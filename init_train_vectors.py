import os
import glob
import torch
import pickle
import jsonlines
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel

from src.utils import DEVICE
from src.normalize_text import normalize


lst_vectorize = [
    ('2wiki', 'train'), ('2wiki', 'eval'),
    ('musique', 'eval'), ('hotpotqa', 'train'),
]


if __name__ == '__main__':
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, normalize_embeddings=True, device=DEVICE)

    for dataset_name, dataset_type in lst_vectorize:
        if dataset_type == "train":
            file_path = f'./train_data/{dataset_name}_train_processed.jsonl'
        elif dataset_type == "eval":
            file_path = f'./eval_data/{dataset_name}_dev_processed.jsonl'

        df_dataset = pd.read_json(file_path, lines=True, orient="records")

        i = df_dataset.shape[0]
        # since python 3.6, default dictionary has been ordered
        d_ctxs = {}
        for ctxs in df_dataset["ctxs"]:
            for ctx in ctxs:
                key = f"Title: {ctx['title']}\nContext: {ctx['text']}"
                if key in d_ctxs:
                    ctx["idx"] = d_ctxs[key]
                else:
                    # ctx_id = s_format_ctx_id.format(id=i)
                    ctx["idx"] = i
                    d_ctxs[key] = i
                    i += 1

        lst_ctx = list(d_ctxs.keys())
        lst_questions_ctxs = df_dataset["question"].to_list() + lst_ctx
        lst_questions_ctxs = list(map(normalize, lst_questions_ctxs))

        # lst_embeddings = []
        for i in tqdm(range(0, len(lst_questions_ctxs), 10240), desc=f"{dataset_name} {dataset_type}"):
            embeddings = model.encode(
                lst_questions_ctxs[i: i + 10240],
                batch_size=8,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            getattr(torch, DEVICE).empty_cache()

            np.save(
                f"embedding_data/{dataset_name}/{dataset_type}_dense{i}.npy",
                embeddings["dense_vecs"]
            )
            # lst_embeddings.append(embeddings)

        input_paths = glob.glob(f"embedding_data/{dataset_name}/{dataset_type}_dense[0-9]*.npy")
        input_paths = sorted(input_paths, key=lambda path: int(path.split("_dense")[1].rstrip(".npy")))
        lst_embeddings = []
        for path in input_paths:
            if not path.startswith(f"embedding_data/{dataset_name}/{dataset_type}_dense") or not path.endswith(".npy"):
                continue

            lst_embeddings.append(np.load(path))

        all_embeddings = np.concatenate(lst_embeddings, axis=0)

        os.makedirs(f"embedding_data/{dataset_name}", exist_ok=True)

        np.save(
            f"embedding_data/{dataset_name}/{dataset_type}_dense.npy",
            all_embeddings)
        np.save(
            f"embedding_data/{dataset_name}/{dataset_type}_content_dense.npy",
            all_embeddings[-len(lst_ctx):])

        for path in input_paths:
            if path.startswith(f"embedding_data/{dataset_name}/{dataset_type}_dense") or not path.endswith(".npy"):
                os.remove(path)

        # save corresponding context array indices
        df_dataset.to_json(file_path, lines=True, orient="records")

        d_ctx_idx = [{"id": i,
                        "title": k.split("\nContext: ")[0].replace("Title: ", ''),
                        "text": k.split("\nContext: ")[1]}
                        for i, k in enumerate(d_ctxs.keys())]

        dataset_path = os.path.join("embedding_data", dataset_name, f"{dataset_type}_passages.jsonl")
        with jsonlines.open(dataset_path, "w") as f:
            f.write_all(d_ctx_idx)
