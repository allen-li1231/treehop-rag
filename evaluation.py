import os
os.environ["HF_HUB_OFFLINE"] = '1'
os.environ["DGLBACKEND"] = "pytorch"
import argparse
import pickle
import functools
from collections.abc import Iterable
import pandas as pd

import src
from src.utils import DEVICE
from src.language_models import MODEL_DICT
from passage_retrieval import MultiHopRetriever
from tree_hop.model import TreeHopBaseModel
from tree_hop.model import TreeHopModel
from tree_hop.asset_loader import get_dataset, get_tree_hop_model


if DEVICE == "mps":
    DEVICE = "cpu"


@functools.lru_cache(maxsize=4)
def get_embedding_model(embedding_name: str, device: str):
    """Load and cache embedding models to avoid re-loading the same model multiple times.
    
    Args:
        embedding_name: Name of the embedding model (e.g., "bge-m3")
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if embedding_name not in MODEL_DICT:
        raise ValueError(f"Unknown embedding name: {embedding_name}, "
                        f"available options are: {list(MODEL_DICT.keys())}")
    
    embedding_model, embedding_tokenizer, _ = src.load_retriever(
        MODEL_DICT[embedding_name]
    )
    embedding_model.eval()
    embedding_model = embedding_model.to(device)
    
    return embedding_model, embedding_tokenizer


@functools.lru_cache()
def get_retriever(
    dataset_name: str,
    embedding_name: str,
    embedding_backend: str,
    model: TreeHopBaseModel | Iterable[TreeHopBaseModel],
    indexing_batch_size=1024,
    index_device=DEVICE,
    embedding_model=None,
    embedding_tokenizer=None,
    **backend_kwargs
):
    """Create MultiHopRetriever for given dataset and models
    
    Args:
        embedding_model: Pre-loaded embedding model (optional)
        embedding_tokenizer: Pre-loaded embedding tokenizer (optional)
    """
    if embedding_name not in MODEL_DICT:
        raise ValueError(f"Unknown embedding name: {embedding_name},"
                         f" available options are: {list(MODEL_DICT.keys())}")

    # check if is iterable of models, if not, convert to tuple
    if not isinstance(model, Iterable):
        model = (model,)

    x_size = next(iter(model)).x_size
    assert all(isinstance(m, TreeHopBaseModel) for m in model), \
        f"Models must be PyTorch modules, got {[type(m) for m in model]}"
    assert all(m.x_size == x_size for m in model), \
        f"All models must have the same x_size, got {[m.x_size for m in model]}"

    data_dir = f"embedding_data/{dataset_name}"

    retriever = MultiHopRetriever(
        MODEL_DICT[embedding_name],
        passages=f"{data_dir}/eval_passages.jsonl",
        passage_embeddings=f"{data_dir}/eval_content_dense.npy",
        backend=embedding_backend,
        save_or_load_index=True,
        faiss_index=f"{data_dir}/",
        projection_size=x_size,
        tree_hop_model=model,
        indexing_batch_size=indexing_batch_size,
        index_device=index_device,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
        **backend_kwargs
    )
    return retriever


def evaluate_retrieve(
    model,
    n_hop,
    top_n,
    prune_redundant=True,
    prune_layer_top=True,
    index_batch_size=1024,
    generate_batch_size=1024,
    embedding_name="bge-m3",
    embedding_backend="hf"
):
    """Evaluate model across multiple datasets with cached embedding model loading.
    
    Args:
        model: The embedding rewriter model to evaluate
        n_hop: Number of hops for multihop retrieval
        top_n: Number of retrieved chunks for each hop
        index_batch_size: Batch size for Faiss retrieval
        generate_batch_size: Batch size for model inference
        embedding_name: Name of the embedding model to use
        embedding_backend: Backend for embedding retrieval (hf or vllm)
        prune_redundant: Whether to prune redundant passages
        prune_layer_top: Whether to prune top layers

        
    Returns:
        Dictionary mapping dataset names to their evaluation statistics
    """
    # Load embedding model using cached function to avoid re-loading the same model
    # The get_embedding_model function is cached, so the same model will be reused
    # across multiple calls to evaluate_retrieve
    embedding_model, embedding_tokenizer = get_embedding_model(embedding_name, DEVICE)

    datasets = ["hotpotqa_distractor", "musique", "multihop_rag"]
    d_stats = {}
    for dataset_name in datasets:
        stat = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            embedding_name=embedding_name,
            embedding_backend=embedding_backend,
            n_hop=n_hop,
            top_n=top_n,
            prune_redundant=prune_redundant,
            prune_layer_top=prune_layer_top,
            index_batch_size=index_batch_size,
            generate_batch_size=generate_batch_size,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer
        )
        d_stats[dataset_name] = stat

    return d_stats


def match_retrieve(df, retrieved_passages):
    set_title = df["set_evidence_title"].copy()
    idx_result = df.name

    lst_match = [0] * len(retrieved_passages)
    for i_hop, retrieved in enumerate(retrieved_passages):
        passage = retrieved[idx_result]
        for psg in passage:
            if psg["title"] not in set_title:
                continue

            set_title.remove(psg["title"])
            lst_match[i_hop] += 1

    return lst_match


def compute_mrr(df_row, retrieved_passages):
    """Compute MRR for a single query over all hops combined.

    Builds a flat deduplicated ranked list (hop order, then within-hop rank)
    and returns 1/rank of the first relevant passage, or 0 if none found.
    """
    evidence_titles = set(df_row["set_evidence_title"])
    idx = df_row.name

    seen = set()
    ranked = []
    for hop_passages in retrieved_passages:
        for psg in hop_passages[idx]:
            title = psg["title"]
            if title not in seen:
                seen.add(title)
                ranked.append(title)

    # calculate MRR based on the ranked list of retrieved passages
    # for hop_passages in retrieved_passages:
    #     for rank, psg in enumerate(hop_passages[idx]):
    #         title = psg["title"]
    #         if title not in seen:
    #             seen.add(title)
    #             if title in evidence_titles:
    #                 ranked.append(1.0 / (rank + 1))  # ranks are 1-indexed for MRR
    # return sum(ranked) / len(evidence_titles) if evidence_titles else 0.0

    for rank, title in enumerate(ranked, start=1):
        if title in evidence_titles:
            return 1.0 / rank
    return 0.0


def complete_chain_recall(df_match, n_hop):
    """Fraction of queries for which *all* gold supporting passages are retrieved.

    Unlike the micro-averaged Recall@K reported elsewhere, which credits a query
    for every gold passage it recovers, this metric is all-or-nothing per query.
    Returned cumulatively, so column i is the fraction of queries whose full
    evidence chain is complete by the end of hop i+1.
    """
    n_evidence = df_match["set_evidence_title"].map(len)
    cum_match = df_match[list(range(n_hop))].cumsum(axis=1)
    return cum_match.ge(n_evidence, axis=0)


def evaluate_dataset(
    model,
    dataset_name,
    embedding_name,
    embedding_backend,
    n_hop,
    top_n,
    prune_redundant=True,
    prune_layer_top=True,
    index_batch_size=10240,
    generate_batch_size=1024,
    embedding_model=None,
    embedding_tokenizer=None
):
    df_QA = get_dataset(dataset_name)
    lst_questions = df_QA["question"].to_list()

    retriever = get_retriever(
        dataset_name=dataset_name,
        embedding_name=embedding_name,
        embedding_backend=embedding_backend,
        model=model.eval(),
        index_batch_size=index_batch_size,
        index_device=DEVICE,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )
    # retriever.search_passages(["Who is "])

    retrieved_result = retriever.multihop_search_passages(
        lst_questions,
        n_hop=n_hop,
        top_n=top_n,
        prune_redundant=prune_redundant,
        prune_layer_top=prune_layer_top,
        index_batch_size=index_batch_size,
        generate_batch_size=generate_batch_size,
    )

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_result.passage,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    n_total = df_match["set_evidence_title"].map(len).sum()

    return df_match[1].sum(axis=0) / n_total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TreeHop")

    # multihop retrieval with TreeHop
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hotpotqa_distractor",
                 "2wiki", "musique", "multihop_rag"],
    )
    
    parser.add_argument(
        "--embedding_name", type=str, default="bge-m3",
        help="Name of the embedding to be used"
    )
    parser.add_argument(
        "--embedding_backend", type=str, default="hf",
        help="Backend for embedding retrieval, supports hf or vllm"
    )
    parser.add_argument(
        "--n_hop", type=int, required=True,
        help="Number of hops for multihop retrieval"
    )
    parser.add_argument(
        "--top_n", type=int, required=True,
        help="Number of retrieved chunks for each hop"
    )

    # Model and retrieval settings
    parser.add_argument(
        "--model_name_or_path", nargs='*',
        default=["allen-li1231/treehop-rag"],
        help="Resume with saved parameters"
    )
    parser.add_argument(
        "--revision", type=str,
        default="main",
        help="Branch name or tag name of the model to be loaded"
    )
    parser.add_argument(
        "--index_batch_size", type=int, default=10240,
        help="Batch size for Fiass retrieval"
    )
    parser.add_argument(
        "--generate_batch_size", type=int, default=2048,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--redundant_pruning", type=bool, default=True,
        help="Toggle stop criterion: redundancy pruning"
    )
    parser.add_argument(
        "--layerwise_top_pruning", type=bool, default=True,
        help="Toggle stop criterion: layer-wise top pruning"
    )
    parser.add_argument(
        "--layerwise_top_pruning_val", type=int, default=None,
        help="Set value of layer-wise top pruning"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(f"Using {DEVICE}")

    args = parse_args()

    print(f"Evaluating {args.dataset_name} with recall@{args.top_n} under {args.n_hop} hops ")
    print(f"Prune redundant: {args.redundant_pruning}, prune layer top: {args.layerwise_top_pruning}")
    lst_tree_hops = []
    for model_path in args.model_name_or_path:
        model = get_tree_hop_model(
            model_path, TreeHopModel,
            device=DEVICE, revision=args.revision,
        )
        lst_tree_hops.append(model)

    df_QA = get_dataset(args.dataset_name)
    retriever = get_retriever(
        dataset_name=args.dataset_name,
        embedding_name=args.embedding_name,
        embedding_backend=args.embedding_backend,
        model=tuple(lst_tree_hops)
    )
    prune_layer_top = args.layerwise_top_pruning_val \
                      if args.layerwise_top_pruning_val is not None \
                      else args.layerwise_top_pruning
    retrieved_result = retriever.multihop_search_passages(
        df_QA["question"].to_list(),
        n_hop=args.n_hop,
        top_n=args.top_n,
        index_batch_size=args.index_batch_size,
        generate_batch_size=args.generate_batch_size,
        prune_redundant=args.redundant_pruning,
        prune_layer_top=prune_layer_top,
        return_tree=True,
        return_query_similarity=True
    )

    retrieved_passages = retrieved_result.passage
    if isinstance(retrieved_result.passage[0][0], dict):
        # For the case of single-hop retrieval
        retrieved_passages = [retrieved_result.passage]

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_passages,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    # df_match = df_match[~df_match["type"].isin(["comparison", "null_query"])]
    n_total = df_match["set_evidence_title"].map(len).sum()

    print("Iteration recalls:")
    print(df_match[range(args.n_hop)].sum(axis=0).cumsum() / n_total)

    df_full_chain = complete_chain_recall(df_match, args.n_hop)
    print("\nComplete evidence-chain recall (all gold passages retrieved, cumulative):")
    print(df_full_chain.mean(axis=0))
    print(f"Avg. gold passages per query: {df_match['set_evidence_title'].map(len).mean():.2f}")
    print("Complete evidence-chain recall by question type:")
    print(pd.concat([df_match["type"], df_full_chain], axis=1)
          .groupby("type")[list(range(args.n_hop))].mean())

    print("\nStats by question type:")
    print(
        df_match.groupby(["type", ])[list(range(args.n_hop))].agg(["count", "mean"])
    )

    mrr_scores = df_QA.apply(
        compute_mrr, retrieved_passages=retrieved_passages, axis=1
    )
    print(f"\nMRR: {mrr_scores.mean():.4f}")
    print("MRR by question type:")
    print(pd.concat([df_QA["type"], mrr_scores.rename("mrr")], axis=1)
          .groupby("type")["mrr"].mean())

    k = 0.
    for i, psgs in enumerate(retrieved_passages):
        k += sum(map(len, psgs))
        print(f"Avg. K on hop {i+1}:", k / len(psgs) if len(psgs) > 0 else 0)

    with open(f"eval_data/{args.dataset_name}_result__top_n={args.top_n}&n_hop={args.n_hop}&redundant={args.redundant_pruning}&layerwise_top={prune_layer_top}.pkl", "wb") as f:
        pickle.dump(retrieved_result, f)