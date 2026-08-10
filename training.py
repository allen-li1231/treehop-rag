import os
os.environ["HF_HUB_OFFLINE"] = '1'
os.environ["DGLBACKEND"] = "pytorch"
import argparse
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm

import random
import numpy as np
import dgl
import torch
import torch.multiprocessing as mp
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Subset
from info_nce import info_nce
from tree_hop.dataset import TreeHopTrainDataset
from tree_hop.model import TreeHopModel
from utils import NodeType

from src.utils import DEVICE
from evaluation import evaluate_dataset, evaluate_retrieve


InBatch = namedtuple(
    "Graph_Reranker_Batch",
    ["graph",]
)


def collate_batch(batch, device: str | torch.device | None = None):
    g = dgl.batch(batch)
    if device is not None:
        g = g.to(device)

    return InBatch(graph=g)


def seed_env(seed: int, device):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not device == "cpu":
        getattr(torch, device).manual_seed(seed)


def seed_worker(worker_id, device):
    worker_seed = torch.initial_seed() % 2**3
    seed_env(worker_seed, device=device)


def gather_graph_contrastive_losses(
    g: dgl.DGLGraph,
    num_negatives,
    loss="infonce",
    negative_mode="paired",
    temperature=0.1
):
    y = g.ndata["y"]
    b_positives = (y == NodeType.relevant_doc.value) | (y == NodeType.leaf.value)
    b_negatives = y == NodeType.irrelevant_doc.value

    idx_out_nodes, idx_in_nodes = g.edges()
    idx_positives = b_positives.nonzero().flatten() # type: ignore
    idx_query_nodes = idx_out_nodes[torch.isin(idx_in_nodes, idx_positives)]

    queries = g.ndata["h"][idx_query_nodes]
    positives = g.ndata["rep"][b_positives]
    negatives = g.ndata["rep"][b_negatives]
    if negative_mode == "paired":
        negatives = negatives.view(positives.size()[0], -1, positives.size()[1])
        if negatives.shape[1] < num_negatives:
            raise LookupError("number of negatives less than specified")

        idx_picked_negs = torch.randint(
            negatives.shape[1], (num_negatives,), device=g.device
        )
        negatives = negatives[:, idx_picked_negs, :]
    elif negative_mode == "unpaired":
        idx_picked_negs = torch.randint(
            negatives.shape[0], (num_negatives,), device=g.device
        )
        negatives = negatives[idx_picked_negs, :]
    else:
        raise NotImplementedError()

    if loss == "infonce":
        loss_func = info_nce
    else:
        raise NotImplementedError()

    loss = loss_func(
        query=queries,
        positive_key=positives,
        negative_keys=negatives,
        temperature=temperature,
        negative_mode=negative_mode,
    )
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graph Reranker")

    parser.add_argument(
        "--device", type=str, default=DEVICE,
        help="Training device (e.g., 'cpu', 'cuda')"
    )

    parser.add_argument(
        "--trainset_name", type=str, default="all",
        help="Name of the training set to be used"
    )
    parser.add_argument(
        "--negative_dataset", type=str, default="hotpotqa_distractor",
        help="Dataset whose corpus supplies the negative pool for contrastive learning"
    )
    parser.add_argument(
        "--embedding_name", type=str, default="bge-m3",
        help="Name of the embedding to be used"
    )
    parser.add_argument(
        "--embedding_backend", type=str, default="hf",
        help="Backend for embedding retrieval, support 'hf' (HuggingFace) and 'vllm'"
    )
    parser.add_argument(
        "--graph_cache_dir", type=str, default=None,
        help="Load dgl graph cache and respective dataset"
    )
    parser.add_argument(
        "--model_cls", type=str,
        default="infonce",
        choices=["infonce"],
        help="Model architecture"
    )
    parser.add_argument(
        "--state_dict", type=str, default=None,
        help="Resume with saved parameters"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoint/",
        help="Directory to save model checkpoints after each epoch"
    )
    parser.add_argument(
        "--sample_size", type=float, default=None,
        help="Whether to sample the training set using a 0-1 float number."
    )
    parser.add_argument(
        "--loss", type=str, default="infonce",
        choices=['infonce'],
        help="Type of loss for contrastive learning"
    )
    parser.add_argument(
        "--n_neg", type=int, default=5,
        help="Number of negatives for each positive sample for contrastive learning"
    )
    parser.add_argument(
        "--neg_mode", type=str, default="paired",
        choices=['paired', 'unpaired'],
        help="Type of negatives w.r.t query for Info NCE loss"
    )
    parser.add_argument(
        "--x_size", type=int, default=1024,
        help="Input size"
    )
    parser.add_argument(
        "--g_size", type=int, default=2048,
        help="Gate size"
    )
    parser.add_argument(
        "--mlp_size", type=int, default=2048,
        help="MLP layer size"
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3,
        help="Number of sequential MLP layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=1,
        help="Number of attention heads per gate"
    )
    parser.add_argument(
        "--n_layer", type=int, default=4,
        help="Number of stacked update blocks"
    )
    parser.add_argument(
        "--norm", type=str, default="rms",
        choices=["rms", "layer", "none"],
        help="Normalisation between stacked attentions, unused when n_layer is 1"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--epoch", type=int, default=4,
        help="Number of training epoch"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Number of training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Training learning rate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.15,
        help="Loss temperature"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=8e-8,
        help="Training weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    mp.set_start_method("spawn")
    # mp.freeze_support()  # For Windows support

    # hyper parameters
    args = parse_args()

    if args.device == "mps":
        # dgl does not support mps on mac m1 chips
        args.device = "cpu"

    print(f"Using {args.device}")

    if isinstance(args.seed, int):
        seed_env(args.seed, device=args.device)

    # create datasets
    train_set = TreeHopTrainDataset(
        args.embedding_name, args.trainset_name, "train",
        num_negatives=args.n_neg,
        negative_dataset=f"embedding_data/{args.negative_dataset}/train_dense.npy",
        graph_cache_dir=args.graph_cache_dir,
        # mp_context=mp.get_context("spawn"),
    )
    if args.sample_size is not None:
        sample_idxs = np.random.choice(
            len(train_set),
            size=args.sample_size,
            replace=False,
        )
        train_set = Subset(train_set, sample_idxs)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=partial(collate_batch, device=args.device),
        shuffle=True,
        num_workers=0,
    )

    # create the model
    if args.model_cls == "infonce":
        model_cls = TreeHopModel
    else:
        raise NotImplementedError(f"Model class '{args.model_cls}' not recognized")

    model = model_cls(
        x_size=args.x_size,
        g_size=args.g_size,
        mlp_size=args.mlp_size,
        n_mlp=args.n_mlp,
        dropout=args.dropout,
        n_head=args.n_head,
        n_layer=args.n_layer,
        norm=args.norm
    )
    if args.state_dict is not None:
        pt_state_dict = torch.load(args.state_dict, weights_only=True, map_location=args.device)
        model.load_state_dict(pt_state_dict)
        print(f"Model checkpoint '{args.state_dict}' loaded")

    print(model.to(args.device))

    # create the optimizer
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # train_loader = eval_loader
    # training loop
    epoch_train_loss = []
    epoch_eval_loss = []
    epoch_eval_score = []
    for epoch in (epoch_pbar:=tqdm(range(1, args.epoch + 1),
                                   desc="Epoch",
                                   position=0,
                                   leave=True)):
        model.train()
        for step, batch in enumerate(pbar:=tqdm(train_loader,
                                                desc="Train step",
                                                position=1,
                                                leave=True,
                                                mininterval=.5)):
            g = batch.graph
            optimizer.zero_grad()
            h = model(g)
            loss = gather_graph_contrastive_losses(
                g, loss=args.loss,
                negative_mode=args.neg_mode,
                num_negatives=args.n_neg, temperature=args.temperature
            )
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3f}"}, refresh=False)

        print(f"Epoch {epoch} avg. loss: {np.mean(epoch_train_loss):.3f}")
        model.eval()
        d_stats = evaluate_retrieve(
            embedding_name=args.embedding_name,
            embedding_backend=args.embedding_backend,
            model=model,
            n_hop=2,
            top_n=5,
            prune_redundant=True,
            prune_layer_top=True,
            generate_batch_size=10240,
        )
        epoch_pbar.set_postfix(d_stats)

        str_eval_stats = '&'.join([f'{k[:8]}={v:.3f}' for k, v in d_stats.items()])
        str_args = '&'.join([f'{k}={v}'
                             for k, v in args.__dict__.items()
                             if k in (
                                 "x_size", "g_size", "mlp_size", "n_mlp", "n_head", "n_layer", "norm",
                                 "lr",
                                 "loss", "n_neg", #"neg_mode",
                                 "seed"
                            )])
        torch.save(
            model.state_dict(),
            f"{args.checkpoint_dir}/{args.model_cls}_{str_eval_stats}__epoch={epoch}&{str_args}.pt",
        )

        epoch_train_loss.clear()
        epoch_eval_loss.clear()
        epoch_eval_score.clear()

        model.reset_query()

    # python training.py --graph_cache_dir train_data/ --loss infonce --temperature 0.08 --n_neg 5 --seed 1307