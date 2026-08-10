import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from FlagEmbedding import BGEM3FlagModel
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["http_proxy"] = "http://127.0.0.1:10809"
# os.environ["https_proxy"] = "http://127.0.0.1:10809"

from src.normalize_text import normalize


if __name__ == '__main__':
    file_path = "eval_data/"
    dataset_name = "multihop_rag"
    dataset_type = "train" if "train_data/" in file_path else "eval"

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, normalize_embeddings=True, device="cuda")
    df_QA = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/MultiHopRAG.json")
    df_dataset = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/corpus.json")

    srs_content: pd.Series = df_dataset.progress_apply(
        lambda df: (f"Title: {df['title']}\n" if df['title'] is not None else "") \
                + (f"Source: {df['source']}\n" if df['source'] is not None else "") \
                + f"Context: {df['body']}",
        axis=1
    )
    srs_content = srs_content.apply(normalize)
    lst_query = df_QA["query"].apply(normalize)

    query_embeddings = model.encode(
        df_QA["query"].to_list(),
        batch_size=32,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    content_embeddings = model.encode(
        srs_content.to_list(),
        batch_size=32,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    os.makedirs(f"embedding_data/{dataset_name}/", exist_ok=True)
    np.save(
        f"embedding_data/{dataset_name}/{dataset_type}_query_dense.npy",
        query_embeddings["dense_vecs"]
    )
    np.save(
        f"embedding_data/{dataset_name}/{dataset_type}_content_dense.npy",
        content_embeddings["dense_vecs"]
    )

    # save corresponding context array indices
    df_dataset["id"] = df_dataset.index
    df_dataset["text"] = df_dataset["body"]

    dataset_path = os.path.join("embedding_data", dataset_name, f"{dataset_type}_passages.jsonl")
    df_dataset[["id", "title", "text"]].to_json(dataset_path, lines=True, orient="records")
