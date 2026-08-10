# mypy: allow-untyped-defs
import os
import argparse
import time
import glob
import itertools

import torch
import torch.nn.functional as F
import faiss
import numpy as np
from tqdm.auto import tqdm
from collections import namedtuple
from typing import Union, List, Iterable

import src.data, src.index, src.slurm, src.normalize_text
from src.utils import DEVICE
from utils import load_file_jsonl, save_file_jsonl

from tree_hop.graph import TreeHopGraph
from tree_hop.model import TreeHopBaseModel, TreeHopModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"


search_passage_results = namedtuple(
    "search_passage_results",
    fields := ["passage", "score", "query_embedding", "passage_embedding"],
    defaults=(None,) * len(fields)
)

multihop_search_passage_results = namedtuple(
    "multihop_search_passage_results",
    fields := ["passage", "tree_hop_graph", "query_similarity"],
    defaults=(None,) * len(fields)
)


class Retriever:
    def __init__(self,
        model_name_or_path: str,
        passages: str,
        passage_embeddings: str | None = None,
        faiss_index: str | None = None,
        backend: str = "hf",
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu",
        # Pre-loaded embedding model (optional)
        embedding_model=None,
        embedding_tokenizer=None,
        # Backend-specific arguments (for backward compatibility)
        no_fp16=None,
        per_gpu_batch_size=None,
        query_maxlength=None,
        **backend_kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.passages = passages
        self.passage_embeddings = passage_embeddings
        self.faiss_index = faiss_index
        if passage_embeddings is None and faiss_index is None:
            raise ValueError("Either passage_embeddings or faiss_index must be provided")

        # Store pre-loaded model if provided
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer

        self.backend = backend.lower()
        if self.backend not in ["hf", "vllm"]:
            raise ValueError(f"Unsupported backend: {backend}. Choose 'hf' or 'vllm'")

        self.save_or_load_index = save_or_load_index
        self.indexing_batch_size = indexing_batch_size
        self.lowercase = lowercase
        self.normalize_text = normalize_text
        self.projection_size = projection_size
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.index_device = index_device

        # Merge backend-specific arguments into backend_kwargs
        # Set defaults if not provided
        if no_fp16 is not None:
            backend_kwargs.setdefault("no_fp16", no_fp16)
        else:
            backend_kwargs.setdefault("no_fp16", False)

        if per_gpu_batch_size is not None:
            backend_kwargs.setdefault("per_gpu_batch_size", per_gpu_batch_size)
        else:
            backend_kwargs.setdefault("per_gpu_batch_size", 64)

        if query_maxlength is not None:
            backend_kwargs.setdefault("query_maxlength", query_maxlength)
        else:
            backend_kwargs.setdefault("query_maxlength", 1024)

        self.backend_kwargs = backend_kwargs

        # Extract commonly used backend kwargs for convenience
        self.no_fp16 = self.backend_kwargs.get("no_fp16", False)
        self.per_gpu_batch_size = self.backend_kwargs.get("per_gpu_batch_size", 64)
        self.query_maxlength = self.backend_kwargs.get("query_maxlength", 1024)

        self.setup_retriever()

    @torch.no_grad
    def embed_queries(self, queries):
        if self.backend == "vllm":
            return self._embed_queries_vllm(queries)
        else:
            return self._embed_queries_hf(queries)

    def _embed_queries_hf(self, queries):
        """Embedding using HuggingFace model"""
        assert isinstance(self.model, torch.nn.Module), \
            "Model must be a HuggingFace model for 'hf' backend"

        embeddings, batch_query = [], []
        for k, q in enumerate(queries):
            if self.lowercase:
                q = q.lower()
            if self.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_query.append(q)

            if len(batch_query) == self.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_query,
                    return_tensors="pt",
                    max_length=self.query_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                output = self.model(**encoded_batch)
                if hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                    output = self._last_token_pool(output, encoded_batch["attention_mask"])

                embeddings.append(output)

                batch_query.clear()
                # getattr(torch, DEVICE).empty_cache()

        embeddings = torch.cat(embeddings, dim=0).to(self.index_device)
        return F.normalize(embeddings, p=2, dim=-1)

    def _embed_queries_vllm(self, queries):
        """Embedding using vLLM model"""
        # Prepare queries with normalization
        processed_queries = []
        for q in queries:
            if self.lowercase:
                q = q.lower()
            if self.normalize_text:
                q = src.normalize_text.normalize(q)
            processed_queries.append(q)

        # vLLM's embed method handles batching internally
        outputs = self.model.embed(processed_queries)
        # Each o.outputs.embedding is a 1-D vector, so these must be stacked into
        # (n_queries, dim). np.concatenate would flatten them into (n_queries * dim,).
        embeddings = np.stack([o.outputs.embedding for o in outputs], axis=0)

        # Convert to torch tensor and move to index_device
        embeddings = torch.from_numpy(embeddings).to(self.index_device)
        return embeddings

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]


    def index_encoded_data(self, input_paths, indexing_batch_size):
        input_paths = sorted(glob.glob(input_paths))
        all_ids = []
        all_embeddings = []
        start_idx = 0

        print(f"Indexing passages from files {input_paths}")
        # start_time_indexing = time.time()
        for i, file_path in enumerate(input_paths):
            data = src.data.load_regular_data(file_path)
            if isinstance(data, tuple):
                ids, embeddings = data
            else:
                embeddings = data
                if embeddings is None:
                    raise ValueError(f"Embeddings in {file_path} is None")

                ids = list(range(start_idx, start_idx + len(embeddings)))
                start_idx += len(embeddings)

            all_ids.extend(ids)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        while all_embeddings.shape[0] > 0:
            all_embeddings, all_ids = self._batch_add_embeddings(
                all_embeddings, all_ids, indexing_batch_size
            )

        # print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    def _batch_add_embeddings(self, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.indexer.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        lst_docs = []
        for passage_ids, scores in top_passages_and_scores:
            lst_doc = []
            for p_id, score in zip(passage_ids, scores):
                doc = passages[p_id].copy()
                doc["score"] = float(score)
                lst_doc.append(doc)

            lst_docs.append(lst_doc)

        return lst_docs

    def _setup_hf_model(self):
        """Setup HuggingFace model with optional tensor parallelism"""
        # print(f"Loading HuggingFace model from: {self.model_name_or_path}")

        self.model, self.tokenizer, _ = src.load_retriever(
            self.model_name_or_path
        )

        # Apply additional kwargs if provided (excluding already handled keys)
        for key, value in self.backend_kwargs.items():
            if key not in ["tensor_parallel_size", "no_fp16", "per_gpu_batch_size", "query_maxlength"] \
                and hasattr(self.model, key):
                setattr(self.model, key, value)

        self.model.eval()
        self.model = self.model.to(DEVICE)
        if not self.no_fp16:
            self.model = self.model.half()

    def _setup_vllm_model(self):
        """Setup vLLM model with tensor parallelism support"""
        # print(f"Loading vLLM model from: {self.model_name_or_path}")
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM is not installed.")

        # vLLM handles device placement and parallelism internally
        vllm_kwargs = {
            # "task": "embed",
            "tensor_parallel_size": self.backend_kwargs.get("tensor_parallel_size", 1)
        }

        # Add dtype configuration
        if not self.no_fp16:
            vllm_kwargs["dtype"] = "half"

        # Merge with user-provided backend_kwargs (excluding already handled keys)
        for key, value in self.backend_kwargs.items():
            if key not in ["tensor_parallel_size", "no_fp16", "per_gpu_batch_size", "query_maxlength"] \
               and key not in vllm_kwargs:
                vllm_kwargs[key] = value

        self.model = LLM(self.model_name_or_path, **vllm_kwargs)
        self.tokenizer = None  # vLLM doesn't expose tokenizer separately

    def setup_retriever(self):
        # Load model based on backend, unless pre-loaded model is provided
        if self.embedding_model is None:
            if self.backend == "vllm":
                self._setup_vllm_model()
            else:
                self._setup_hf_model()
        else:
            # Use pre-loaded model
            self.model = self.embedding_model
            self.tokenizer = self.embedding_tokenizer

        self.indexer = src.index.Indexer(
            self.projection_size, self.n_subquantizers, self.n_bits
        )
        if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
            if src.slurm.is_distributed():
                self.indexer.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), src.slurm.local_rank, self.indexer.index
                )
            else:
                n_gpus = faiss.get_num_gpus()
                if n_gpus <= 0:
                    raise LookupError("Fiass cannot detect a gpu")

                if n_gpus == 1:
                    self.indexer.index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, self.indexer.index
                    )
                else:
                    self.indexer.index = faiss.index_cpu_to_all_gpus(self.indexer.index)

        # index all passages
        input_paths = glob.glob(self.faiss_index or self.passage_embeddings)
        embeddings_dir = os.path.dirname(input_paths[0])
        if isinstance(self.faiss_index, str) and os.path.exists(self.faiss_index):
            self.indexer.deserialize_from(self.faiss_index)
        elif os.path.exists(self.passage_embeddings):
            self.index_encoded_data(self.passage_embeddings, self.indexing_batch_size)
            if self.save_or_load_index:
                if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
                    self.indexer.index = faiss.index_gpu_to_cpu(self.indexer.index)
                self.indexer.serialize(embeddings_dir)
        else:
            raise FileNotFoundError(
                f"Passage embeddings not found at {self.passage_embeddings}, "
                f"or faiss index not found at {self.faiss_index}"
            )

        # load passages
        self.passages = src.data.load_regular_data(self.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        # print(f"{len(self.passages)} passages have been loaded")

    def get_passage_embedding_by_id(self, passage_ids):
        if isinstance(passage_ids, int):
            return self.indexer.index.reconstruct(passage_ids)

        passage_embedding = []
        for p_id in passage_ids:
            passage_embedding.append(self.indexer.index.reconstruct(p_id))

        return passage_embedding

    def search_passages(
        self,
        query: Union[str, Iterable[str], torch.Tensor, np.ndarray],
        top_n=10,
        index_batch_size=2048,
        return_query_embeddings=False,
        return_passage_embeddings=False
    ):
        queries = [query] if isinstance(query, str) else query

        query_embeddings = \
            queries \
            if isinstance(queries, (torch.Tensor, np.ndarray)) \
            else self.embed_queries(queries)

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()

        # get top k results
        top_ids_and_scores = self.indexer.search_knn(
            query_vectors=query_embeddings,
            top_docs=top_n,
            index_batch_size=index_batch_size
        )

        lst_passages = self.add_passages(self.passage_id_map, top_ids_and_scores)

        lst_passage_embeddings = []
        lst_scores = []
        for passage_ids, scores in top_ids_and_scores:
            lst_scores.append(scores)
            passage_embeddings = self.get_passage_embedding_by_id(passage_ids)
            lst_passage_embeddings.append(passage_embeddings)

        score = np.vstack(lst_scores)
        if return_passage_embeddings:
            passage_embedding = (np.vstack(lst_passage_embeddings)
                                 .reshape((query_embeddings.shape[0], top_n, -1)))
        else:
            passage_embedding = None

        return search_passage_results(
            passage=lst_passages,
            score=score,
            query_embedding=query_embeddings if return_query_embeddings else None,
            passage_embedding=passage_embedding
        )


class MultiHopRetriever(Retriever):
    def __init__(
        self,
        model_name_or_path,
        passages,
        passage_embeddings,
        tree_hop_model: TreeHopBaseModel | Iterable[TreeHopBaseModel],
        faiss_index: str | None = None,
        backend: str = "hf",
        no_fp16=True,
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        per_gpu_batch_size=64,
        query_maxlength=512,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu",
        # Pre-loaded embedding model (optional)
        embedding_model=None,
        embedding_tokenizer=None,
        **backend_kwargs
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            passages=passages,
            passage_embeddings=passage_embeddings,
            faiss_index=faiss_index,
            backend=backend,
            no_fp16=no_fp16,
            save_or_load_index=save_or_load_index,
            indexing_batch_size=indexing_batch_size,
            lowercase=lowercase,
            normalize_text=normalize_text,
            per_gpu_batch_size=per_gpu_batch_size,
            query_maxlength=query_maxlength,
            projection_size=projection_size,
            n_subquantizers=n_subquantizers,
            n_bits=n_bits,
            index_device=index_device,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer,
            **backend_kwargs
        )

        self.lst_tree_hop_model = [tree_hop_model] \
                                  if isinstance(tree_hop_model, TreeHopBaseModel) \
                                  else tree_hop_model

    def reset_query(self):
        for tree_hop_model in self.lst_tree_hop_model:
            tree_hop_model.reset_query()

    def multihop_search_passages(
        self,
        query: Union[List[str], str],
        n_hop: int,
        top_n: int = 10,
        min_ranking: int | None = None,
        index_batch_size=10240,
        generate_batch_size=1024,
        show_progress=True,
        prune_redundant=True,
        prune_layer_top: Union[int, bool] = True,
        return_tree=False,
        return_query_similarity=False
    ):
        assert isinstance(n_hop, int) and n_hop > 0, "n_hop must be a positive integer"
        gen_device = self.lst_tree_hop_model[0].device

        pbar = tqdm(
            total=n_hop,
            desc="Retrieving",
            postfix={"num_query": len(query)},
            leave=True,
            disable=not show_progress
        )

        query = [query] if isinstance(query, str) else query
        # start_time_search = time.time()
        search_result = self.search_passages(
            query,
            top_n=top_n,
            index_batch_size=index_batch_size,
            return_query_embeddings=True,
            return_passage_embeddings=True
        )
        # pbar.set_postfix({"num_query": len(query_embeddings),
        #                   "elapsed": time.time() - start_time_search})

        # if n_hop == 1:
        #     pbar.close()
        #     return search_result

        self.reset_query()

        lst_q_emb = []
        for tree_hop_model in self.lst_tree_hop_model:
            q_emb = tree_hop_model.next_query(
                q_emb=search_result.query_embedding,
                ctx_embs=search_result.passage_embedding,
                batch_size=generate_batch_size
            )
            lst_q_emb.append(q_emb)

        q_emb = torch.cat(lst_q_emb, dim=0).to(gen_device)

        pbar.set_postfix({"num_query": q_emb.shape[0],
                        #   "elapsed": time.time() - start_time_generate
                          })

        # (num_queries, top_n, num_dim)
        last_ctx_emb = (torch.from_numpy(search_result.passage_embedding)
                        .to(gen_device))

        last_q_emb = (torch.from_numpy(search_result.query_embedding)
                      .to(gen_device)
                      .unsqueeze(1))

        last_score = F.cosine_similarity(last_q_emb, last_ctx_emb, dim=-1)
        last_score = last_score.cpu().repeat(1, len(self.lst_tree_hop_model)).numpy()

 
        retriever_ids = [[i]*top_n for i in range(len(self.lst_tree_hop_model))]
        retriever_ids = np.ravel(retriever_ids)
        tree_hop_graphs = [TreeHopGraph(q, [psg * len(self.lst_tree_hop_model)],
                                        score=score,
                                        top_n=top_n,
                                        min_ranking=min_ranking,
                                        prune_redundant=prune_redundant,
                                        prune_layer_top=False,
                                        retriever_ids=retriever_ids)
                           for q, psg, score, \
                               in zip(query, search_result.passage,
                                                     last_score,
                                                     strict=True)]

        lst_results = [[graph.filtered_passages for graph in tree_hop_graphs]]
        pbar.update(1)

        if n_hop == 1:
            pbar.close()
            return multihop_search_passage_results(
                passage=lst_results,
                tree_hop_graph=tree_hop_graphs if return_tree else None,
                # query_similarity=query_sims if return_query_similarity else None
            )

        for i_hop in range(1, n_hop):

            query_passage_masks = [
                [graph.get_query_passage_mask_by_retriever_id(retriever_id)
                    for graph in tree_hop_graphs]
                for retriever_id in range(len(self.lst_tree_hop_model))
            ]

            grouped_query_passage_masks = [np.concatenate(m, axis=None)
                                            for m in query_passage_masks]

            lst_q_emb = [
                q[m]
                for q, m in zip(lst_q_emb, grouped_query_passage_masks)
            ]
            q_emb = torch.cat(lst_q_emb, dim=0)
            if q_emb.shape[0] == 0:
                for _ in range(i_hop, n_hop):
                    lst_results.append([])

                break

            pbar.set_description("Retrieving")
            # start_time_search = time.time()
            search_result = self.search_passages(
                q_emb,
                top_n=top_n,
                index_batch_size=index_batch_size,
                return_passage_embeddings=True
            )
            last_q_emb = q_emb

            pbar.set_description("Generating")

            ctx_embs = search_result.passage_embedding.reshape(-1, q_emb.shape[-1])
            # sum over queries for each retriever
            # then multiply by top_n to get number of passages per retriever
            ctx_offsets = [top_n * masks.sum() for masks in grouped_query_passage_masks]

            # start_time_generate = time.time()
            # assume embeddings reconstructed from faiss are normalized before stored
            # filter out semantically distant passage embeddings
            lst_q_emb = []
            ctx_i = 0
            for tree_hop_model, q_p_masks, ctx_offset \
                in zip(self.lst_tree_hop_model,
                       query_passage_masks,
                       ctx_offsets,
                       strict=True):
                if ctx_offset == 0:
                    continue

                q_emb = tree_hop_model.next_query(
                    ctx_embs=ctx_embs[ctx_i: ctx_i + ctx_offset],
                    query_passage_masks=q_p_masks,
                    batch_size=generate_batch_size,
                    top_n=top_n,
                )

                lst_q_emb.append(q_emb)
                ctx_i += ctx_offset

            assert ctx_i == ctx_embs.shape[0], \
                "Some passage embeddings are not assigned to any retriever."

            q_emb = torch.cat(lst_q_emb, dim=0)

            pbar.set_postfix({"num_query": len(q_emb),
                            #   "elapsed": time.time() - start_time_generate
                              })

            last_ctx_emb = (torch.from_numpy(search_result.passage_embedding)
                            .to(gen_device))

            last_q_emb = (last_q_emb
                          .to(gen_device)
                          .repeat(1, top_n)
                          .view_as(last_ctx_emb))
    
            last_score = F.cosine_similarity(last_q_emb, last_ctx_emb, dim=-1)
            last_score = last_score.cpu().numpy()


            lst_passages = []
            # for each retriever, assign retrieved passages to each tree hop graph
            # initiate passage indices for retrievers
            i_retriever_ids = [0] + list(map(len, lst_q_emb))[:-1]
            for graph in tree_hop_graphs:
                num_queries = [graph.get_query_passage_mask_by_retriever_id(r_id).sum()
                                for r_id in range(len(self.lst_tree_hop_model))]

                # take out corresponding passages for each retriever
                passage_layer = list(itertools.chain.from_iterable([
                    search_result.passage[i // top_n: i // top_n + n]
                     for i, n in zip(i_retriever_ids,
                                       num_queries)
                ]))
                score = last_score.ravel()[
                    np.hstack([
                        np.arange(i, i + top_n * n)
                        for i, n in zip(i_retriever_ids, num_queries)
                    ])
                ]
                # q_sim = query_sim[i_current: i_current + num_query].ravel()
                # g_sim = gen_sim[i_current: i_current + num_query].ravel()

                graph.add_passage_layer(
                    passage_layer,
                    top_n=top_n,
                    prune_redundant=prune_redundant,
                    prune_layer_top=prune_layer_top,
                    score=score,
                    # query_sim=q_sim,
                    # gen_sim=g_sim,
                    min_ranking=min_ranking,
                )

                # query_passage_masks.append(graph.query_passage_mask)
                lst_passages.append(graph.filtered_passages)
                i_retriever_ids = [i + top_n * n for i, n in zip(i_retriever_ids, num_queries)]

            assert i_retriever_ids == list(itertools.accumulate(map(len, lst_q_emb))), \
                "Some passages are not assigned to any graph."

            lst_results.append(lst_passages)
            pbar.update(1)
            pbar.set_postfix({"num_query": len(q_emb),
                            #   "elapsed": time.time() - start_time_search
                              })
            if show_progress:
                print(f"Hop {i_hop}/{n_hop} done.")

        pbar.close()

        self.reset_query()

        return multihop_search_passage_results(
            passage=lst_results,
            tree_hop_graph=tree_hop_graphs if return_tree else None,
            # query_similarity=query_sims if return_query_similarity else None
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--passages",
        type=str,
        help="Path to passages (.tsv file)"
    )
    parser.add_argument(
        "--passage_embeddings",
        type=str,
        default=None,
        help="Path to encoded passages in Numpy format"
    )
    parser.add_argument(
        "--faiss_index",
        type=str,
        default=None,
        help="Path to encoded passages in Faiss format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="dir path to save embeddings"
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Id of the current shard"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards"
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="Number of documents to retrieve per questions"
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=64,
        help="Batch size for question encoding"
    )
    parser.add_argument(
        "--save_or_load_index",
        action="store_true",
        help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="inference in fp32")
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a question"
    )
    parser.add_argument(
        "--indexing_batch_size",
        type=int,
        default=1000000,
        help="Batch size of the number of passages indexed"
    )
    parser.add_argument(
        "--projection_size",
        type=int,
        default=768
    )
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=8,
        help="Number of bits per subquantizer"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="lowercase text before encoding"
    )
    parser.add_argument(
        "--normalize_text",
        action="store_true",
        help="normalize text"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards > 1:
        src.slurm.init_distributed_mode(args)

    # for debugging
    # data_paths = glob.glob(args.data)
    retriever = Retriever(
        model_name_or_path=args.model_name_or_path,
        passages=args.passages,
        passage_embeddings=args.passage_embeddings,
        faiss_index=args.faiss_index,
        no_fp16=args.no_fp16,
        save_or_load_index=args.save_or_load_index,
        indexing_batch_size=args.indexing_batch_size,
        lowercase=args.lowercase,
        normalize_text=args.normalize_text,
        per_gpu_batch_size=args.per_gpu_batch_size,
        query_maxlength=args.question_maxlength,
        projection_size=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits
    )

    query = args.query
    if os.path.exists(query):
        query = load_file_jsonl(query)

        shard_size = len(query) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(query)

        query = query[start_idx: end_idx]
        print("query length:", len(query))

    retrieved_documents = retriever.search_passages(query, args.n_docs).passage
    if isinstance(args.output, str):
        if isinstance(query, str):
            data = [{"question": query, "ctxs": retrieved_documents}]
        else:
            data = [{"question": question, "ctxs": ctx}
                    for question, ctx in zip(query, retrieved_documents)]
        save_file_jsonl(data, args.output)
    else:
        print(retrieved_documents)


if __name__ == "__main__":
    # --query "What is the occupation of Obama?" --passages ./wikipedia_data/psgs_w100.tsv --passage_embeddings "./wikipedia_data/embedding_contriever-msmarco/*" --model_name_or_path "facebook/contriever-msmarco" --output ./train_data/extractor_retrieve_wiki.jsonl
    # main()

    DEVICE = "cpu" if DEVICE == "mps" else DEVICE

    def get_dataset(dataset_name):
        import pandas as pd
        df_QA = pd.read_json(f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True)
        df_QA = (df_QA[~df_QA["type"].isin(["comparison", # 2wiki
                                            # multihop_rag
                                            "comparison_query", "null_query", "temporal_query"
                                            ])]
                .reset_index())
        df_QA["set_evidence_title"] = df_QA["supporting_facts"].apply(
            lambda lst: set([evd[0] for evd in lst])
        )
        return df_QA  # for testing

    dataset_name = "2wiki"  # hotpotqa_distractor, 2wiki, musique, multihop_rag

    n_hop=3
    top_n=5
    prune_redundant = True
    prune_layer_top = True
    min_ranking = None

    tree_hop_pt_file = "checkpoint/infonce_hotpotqa=0.055&musique=0.051&multihop=0.093__epoch=2&loss=infonce&n_neg=6&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_layer=3&norm=rms&lr=2e-05&seed=1307.pt"

    from tree_hop.asset_loader import get_tree_hop_model
    tree_hop_model = get_tree_hop_model(tree_hop_pt_file, TreeHopModel, DEVICE)

    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages=f"embedding_data/{dataset_name}/eval_passages.jsonl",
        passage_embeddings=f"embedding_data/{dataset_name}/eval_content_dense.npy",
        tree_hop_model=tree_hop_model,
        projection_size=1024,
        save_or_load_index=True,
        indexing_batch_size=10240,
        index_device=DEVICE
    )
    df_QA = get_dataset(dataset_name)
    lst_questions = df_QA["question"].to_list()
    # lst_questions = ["How many times did plague occur in the birth city of the composer of La fida ninfa?"]

    retrieved_result = retriever.multihop_search_passages(
        lst_questions,
        n_hop=n_hop,
        top_n=top_n,
        prune_redundant=prune_redundant,
        prune_layer_top=prune_layer_top,
        min_ranking=min_ranking
    )

    retrieved_passages = retrieved_result.passage
    if isinstance(retrieved_result.passage[0][0], dict):
        # For the case of single-hop retrieval
        retrieved_passages = [retrieved_result.passage]


    def match_retrieve(df, retrieved_passages):
        set_title = df["set_evidence_title"].copy()
        idx_result = df.name

        lst_match = [0] * len(retrieved_passages)
        for i_hop, retrieved in enumerate(retrieved_passages):
            if len(retrieved) == 0:
                continue

            passage = retrieved[idx_result]
            for psg in passage:
                if psg["title"] not in set_title:
                    continue

                set_title.remove(psg["title"])
                lst_match[i_hop] += 1

        return lst_match


    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_passages,
        axis=1,
        result_type="expand"
    )
    import pandas as pd
    df_match = pd.concat([df_QA, df_match], axis=1)
    # df_match = df_match[~df_match["type"].isin(["comparison", "null_query"])]
    n_total = df_match["set_evidence_title"].map(len).sum()

    print("Iteration recalls:")
    print(df_match[range(n_hop)].sum(axis=0).cumsum() / n_total)

    print("Stats by question type:")
    print(
        df_match.groupby(["type", ])[list(range(n_hop))].agg(["count", "mean"])
    )

    k = 0.
    for i, psgs in enumerate(retrieved_passages):
        k += sum(map(len, psgs))
        print(f"Avg. K on hop {i+1}:", k / len(psgs) if len(psgs) > 0 else 0)
