# mypy: allow-untyped-defs
import os
import re
from typing import Iterable, Union
import dgl
import networkx as nx
import itertools
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import NodeType
from src.normalize_text import normalize
from src.evaluation import normalize_unicode, white_space_fix
from src.concurrency import bounded_process_pool_map


# ---------------------------------------------------------------------------
# Type category sets for graph topology validation
# ---------------------------------------------------------------------------

# Single root types: 1 first-layer node, ≥1 last-layer nodes
SINGLE_ROOT_TYPES = frozenset({
    'compositional', 'inference',  # 2wiki_devMultiHopQA
    'bridge',                       # HotpotQA
    '2hop', '3hop1', '4hop1',       # musique_dev (linear chains)
})

# Comparison types: all first-layer nodes are also last-layer nodes (independent)
COMPARISON_TYPES = frozenset({
    'comparison',  # 2wiki_devMultiHopQA & HotpotQA
})

# Bridge comparison types: multiple first-layer nodes, multiple last-layer nodes
BRIDGE_COMPARISON_TYPES = frozenset({
    'bridge_comparison',  # 2wiki_devMultiHopQA
})

# Convergent types: multiple first-layer nodes converge to ≥1 last-layer nodes
CONVERGENT_TYPES = frozenset({
    '3hop2', '4hop2', '4hop3',  # musique_dev (converging topologies)
})


# Module-level helper functions for multiprocessing compatibility
# These functions avoid pickling the entire TreeHopTrainDataset instance

def _clean_title(title: str) -> str:
    """Clean and normalize a title string."""
    return normalize_unicode(white_space_fix(normalize(title).strip()))


def _match_evidence_title(evidences, supporting_facts, ctxs):
    """Match evidence titles with supporting facts and contexts."""
    is_match = True
    for evi in evidences:
        if evi[0] not in supporting_facts:
            is_match = False
            for fact in supporting_facts:
                if _clean_title(evi[0]) == _clean_title(fact):
                    evi[0] = fact
                    is_match = True

            if not is_match:
                for fact in supporting_facts:
                    if _clean_title(evi[0]) in _clean_title(fact):
                        evi[0] = fact
                        is_match = True

            if not is_match:
                for ctx in ctxs:
                    if evi[0] == ctx["title"] or ctx["text"].startswith(evi[0]):
                        evi[0] = ctx["title"]
                        is_match = True

        if evi[2] not in supporting_facts:
            is_match = False
            cleaned_evi = _clean_title(evi[2])
            for fact in supporting_facts:
                cleaned_fact = _clean_title(fact)
                if cleaned_evi == cleaned_fact:
                    evi[2] = fact
                    is_match = True

            if not is_match:
                for fact in supporting_facts:
                    cleaned_fact = _clean_title(fact)
                    if cleaned_evi.endswith(cleaned_fact) or cleaned_evi.startswith(cleaned_fact) \
                        or cleaned_fact.endswith(cleaned_evi) or cleaned_fact.startswith(cleaned_evi):
                        evi[2] = fact
                        is_match = True

            if not is_match:
                for ctx in ctxs:
                    if evi[2] == ctx["title"] or ctx["text"].startswith(evi[2]):
                        evi[2] = ctx["title"]
                        is_match = True


def _gen_negative_samples(
        num_samples: int,
        num_positives: int,
        num_negatives: int,
        exclude: Union[set, Iterable],
        idx_neg_start: int | None,
        n_total_contexts: int
    ):
    """Generate negative samples for contrastive learning.

    When ``idx_neg_start`` is *None*, negatives are drawn from the local
    context pool ``range(num_samples)``.  When a negative dataset is
    present (``idx_neg_start is not None``), negatives are drawn from
    the **combined** pool ``range(n_total_contexts)`` (main + negative
    dataset), always excluding positive indices supplied via *exclude*.
    """
    if not isinstance(exclude, set):
        exclude = set(exclude)

    if idx_neg_start is None:
        # Local pool: indices into the record's ctxs list
        lst_choices = [n for n in range(num_samples) if n not in exclude]
        lst_negatives: list[list[int]] = []
        for _ in range(num_positives):
            ary_negatives = np.random.choice(
                lst_choices,
                size=num_negatives,
                replace=False
            )
            lst_negatives.append(ary_negatives.tolist())
    else:
        # Combined pool: global embedding indices [0, n_total_contexts)
        # Use rejection sampling (exclude set is tiny relative to pool)
        lst_negatives: list[list[int]] = []
        for _ in range(num_positives):
            neg_set: set[int] = set()
            while len(neg_set) < num_negatives:
                idx = int(np.random.randint(0, n_total_contexts))
                if idx not in exclude:
                    neg_set.add(idx)
            lst_negatives.append(list(neg_set))

    return lst_negatives


def _resolve_title2idx(set_titles, ctxs):
    """Map each evidence source title to a *unique* ctx index.

    Exact title matches are assigned first and reserve their ctx; remaining
    titles fall back to substring containment (``ctx["title"] in title``,
    preferring the longest — i.e. most specific — ctx title) and then to a
    text-prefix relaxation.  Each ctx index is claimed by at most one title.

    The original last-wins dict comprehension merged two source titles onto the
    same ctx whenever a short ctx title happened to be a substring of another
    title (e.g. ctx "Houston" sits inside "The Collegian (Houston Baptist
    University)"), collapsing the graph to a single node.  Exact-match priority
    plus per-ctx reservation removes that collision.
    """
    d_title2idx = {}
    used = set()

    # Pass 1: exact matches win and reserve their ctx index.
    unresolved = []
    for title in set_titles:
        exact = next((n for n, ctx in enumerate(ctxs)
                      if ctx["title"] == title and n not in used), None)
        if exact is not None:
            d_title2idx[title] = exact
            used.add(exact)
        else:
            unresolved.append(title)

    # Pass 2: substring containment (longest ctx title first), then text prefix.
    for title in unresolved:
        cands = [(len(ctx["title"]), n)
                 for n, ctx in enumerate(ctxs)
                 if n not in used
                 and (ctx["title"] in title or ctx["text"].startswith(title))]
        if cands:
            n = max(cands)[1]
            d_title2idx[title] = n
            used.add(n)

    return d_title2idx


def _graph_propagator(row, num_negatives, idx_neg_start, n_total_contexts):
    """Generate indices of positive and negative samples in breadth-first search order."""
    ctxs = row["ctxs"]
    evidences = row["evidences"]
    supporting_facts = {fact[0] for fact in row["supporting_facts"]}

    # MuSiQue terminal edges are [P_last, "", answer]: the target is the *answer*
    # text (empty relation), which is deliberately not a passage title so the last
    # paragraph is treated as a leaf.  _match_evidence_title is built for 2wiki,
    # whose targets are real passages; run on a musique answer it fuzzy-matches the
    # answer back onto its own source paragraph (e.g. "Al Saud" -> "Prince ... Al
    # Saud"), producing a self-edge that trips the DAG check ("Detect loop in the
    # graph").  Only reconcile titles for datasets with real relations.
    is_answer_leaf_style = bool(evidences) and all(
        len(evi) > 1 and evi[1] == "" for evi in evidences
    )
    if not is_answer_leaf_style:
        _match_evidence_title(evidences, supporting_facts, ctxs)

    set_titles = set(src for src, _, dst in evidences)
    d_title2idx = _resolve_title2idx(set_titles, ctxs)

    assert len(d_title2idx) == len(set_titles), f"{set_titles}\n{[ctx['title'] for ctx in ctxs]}"

    d_nodes = dict()
    set_last_layer_nodes = set()
    for evi in evidences:
        idx_src, idx_dst = d_title2idx[evi[0]], d_title2idx.get(evi[2], None)
        if idx_dst is None or idx_dst == idx_src:
            # reaches last layer. idx_dst is None when the target is the answer
            # text (musique terminal edge); idx_dst == idx_src when that answer
            # text happens to coincide with the source passage's own title
            # (e.g. answer "Kuybyshev Reservoir" resolving back to the passage of
            # the same name).  Both mean this node is a leaf, not a self-edge —
            # keeping the self-edge would trip the DAG check below.
            set_last_layer_nodes.add(idx_src)
            if idx_src not in d_nodes:
                d_nodes[idx_src] = []
            continue

        elif idx_src not in d_nodes:
            d_nodes[idx_src] = [idx_dst]
        elif idx_dst not in d_nodes[idx_src]:
            d_nodes[idx_src].append(idx_dst)

    assert len(d_nodes) > 1, "Detect single-hop graph"
    g = nx.DiGraph(d_nodes)
    assert nx.is_directed_acyclic_graph(g), \
        "Detect loop in the graph: the same context has been used more than once."

    # Pre-compute all positive global embedding indices for exclusion
    # when sampling from the combined (main + negative) pool.
    if idx_neg_start is not None:
        all_positive_global = {ctxs[i]["idx"] for i in d_title2idx.values()}

    set_start_nodes = set(d_nodes.keys())
    set_end_nodes = set(itertools.chain(*d_nodes.values()))

    # first_layer nodes
    set_first_layer_nodes = set_start_nodes - set_end_nodes

    query_type = row["type"]
    if query_type in SINGLE_ROOT_TYPES:
        assert len(set_first_layer_nodes) == 1 and len(set_last_layer_nodes) > 0, \
            f"Not a {query_type}"
    elif query_type in COMPARISON_TYPES:
        assert len(set_first_layer_nodes) > 1 and (set_first_layer_nodes == set_last_layer_nodes), \
            f"Not a {query_type}"
    elif query_type in BRIDGE_COMPARISON_TYPES:
        assert len(set_first_layer_nodes) > 1 and len(set_last_layer_nodes) > 1, \
            f"Not a {query_type}"
    elif query_type in CONVERGENT_TYPES:
        assert len(set_first_layer_nodes) > 1 and len(set_last_layer_nodes) > 0, \
            f"Not a {query_type}"

    # BFS
    current: list[list[int]] = [list(set_first_layer_nodes)]
    while any(len(nodes) > 0 for nodes in current):
        lst_current = list(itertools.chain(*current))
        lst_current_negatives: list[list[list[int]]] = []
        for positives in current:
            lst_children = list(itertools.chain(*(
                d_nodes.get(node, [])
                for node in positives)
            ))

            if idx_neg_start is not None:
                exclude = all_positive_global
            else:
                exclude = lst_current + lst_children

            negatives: list[list[int]] = _gen_negative_samples(
                num_positives=len(positives),
                num_samples=len(ctxs),
                num_negatives=num_negatives,
                exclude=exclude,
                idx_neg_start=idx_neg_start,
                n_total_contexts=n_total_contexts
            )
            lst_current_negatives.append(negatives)

        yield current, lst_current_negatives

        current = [d_nodes.get(node, []) for node in lst_current]


def _make_comparison_trainable(
    row, lst_graph_idx, num_negatives, idx_neg_start, n_total_contexts
):
    """For comparison queries, split into two separate linear graphs
    (query → P1 → P2 and query → P2 → P1) instead of cross-connecting
    them in a single graph.

    Negatives are re-sampled **independently** for every branch of every
    variant so that the two orderings see different negative passages.

    Returns a list of graph-index lists. For non-comparison queries the
    list has one element (the original). For comparison queries it has two.
    """
    query_type = row["type"]
    if "comparison" not in query_type or len(lst_graph_idx) != 1:
        return [lst_graph_idx]

    (lst_query_idx,), (lst_ctx_idx,) = lst_graph_idx[0]
    if len(lst_query_idx) != 2:
        return [lst_graph_idx]

    idx_a, idx_b = lst_query_idx
    ctxs = row["ctxs"]

    # Build exclude set: both positives must never appear as negatives
    if idx_neg_start is not None:
        exclude = {ctxs[idx_a]["idx"], ctxs[idx_b]["idx"]}
    else:
        exclude = {idx_a, idx_b}

    def _sample(n_pos=1):
        return _gen_negative_samples(
            num_samples=len(ctxs),
            num_positives=n_pos,
            num_negatives=num_negatives,
            exclude=exclude,
            idx_neg_start=idx_neg_start,
            n_total_contexts=n_total_contexts,
        )

    # Graph 1: query → idx_a → idx_b (leaf)  — fresh negatives
    [neg_a1] = _sample()
    [neg_b1] = _sample()
    graph_1 = [
        ([[idx_a]], [[neg_a1]]),
        ([[idx_b]], [[neg_b1]]),
    ]

    # Graph 2: query → idx_b → idx_a (leaf)  — independently sampled negatives
    [neg_b2] = _sample()
    [neg_a2] = _sample()
    graph_2 = [
        ([[idx_b]], [[neg_b2]]),
        ([[idx_a]], [[neg_a2]]),
    ]
    return [graph_1, graph_2]


def _build_single_graph(index, row, lst_idx_graph, num_negatives, idx_neg_start):
    """Build a single graph's plain-Python data from BFS layer indices."""
    lst_last_layer_pos = []
    idx_current = 1
    lst_y: list[int] = [NodeType.query.value]
    prev_pos = [0]
    lst_node_out = []
    lst_node_in = []
    lst_idx_ctxs = [index]  # query node

    # propagate in BFS order
    for lst_idx_pos, lst_idx_negs in lst_idx_graph:
        assert len(prev_pos) == len(lst_idx_pos) == len(lst_idx_negs), \
            "number of nodes in new layer mismatches with the last layer"
        # layer-wise
        lst_tmp = []
        for pp, idx_pos, idx_negs in zip(prev_pos, lst_idx_pos, lst_idx_negs):
            if len(idx_pos) == 0:
                lst_last_layer_pos.append(pp)
                continue

            lst_y.extend([
                NodeType.relevant_doc.value,
                *([NodeType.irrelevant_doc.value] * num_negatives)
            ] * len(idx_pos))
            # assign node index
            n_current_nodes = len(idx_pos) * (1+num_negatives)
            lst_current = list(range(idx_current, idx_current + n_current_nodes))

            for n, (i_pos, lst_i_negs) in enumerate(zip(idx_pos, idx_negs)):
                lst_node_out.extend([pp] * (1+num_negatives))
                lst_node_in.extend(lst_current[n * (1+num_negatives): (n+1) * (1+num_negatives)])
                if idx_neg_start is None:
                    lst_idx_ctxs.extend([row["ctxs"][idx]["idx"] for idx in [i_pos] + lst_i_negs])
                else:
                    lst_idx_ctxs.extend([row["ctxs"][i_pos]["idx"]] + lst_i_negs)

            lst_tmp.extend(lst_current[:: 1+num_negatives])
            idx_current += n_current_nodes

        prev_pos = lst_tmp

    # label last layer positive nodes to pseudo query node
    lst_last_layer_pos.extend(prev_pos)
    for pos in lst_last_layer_pos:
        lst_y[pos] = NodeType.leaf.value

    return lst_node_out, lst_node_in, lst_y, idx_current, lst_idx_ctxs


def _create_graph_helper(
    index: int,
    row,
    num_negatives: int,
    exclude_comparison: bool,
    idx_neg_start: int | None,
    n_total_contexts: int
):
    """
    Module-level helper function for creating graphs in multiprocessing workers.
    
    This function is picklable and doesn't require the entire TreeHopTrainDataset instance.
    Returns a single graph-data tuple, or a list of tuples for comparison
    queries (one per ordering variant).
    """
    if exclude_comparison and row["type"] == "comparison":
        return index

    try:
        lst_idx_graph = list(_graph_propagator(
            row, num_negatives, idx_neg_start, n_total_contexts
        ))
        graph_variants = _make_comparison_trainable(
            row, lst_idx_graph, num_negatives, idx_neg_start, n_total_contexts
        )
    except AssertionError as e:
        print(f"  [SKIP] Index {index} AssertionError: {e}")
        return index
    except TypeError as e:
        print(f"  [SKIP] Index {index} TypeError: {e} | evidences[0]={row['evidences'][0]}")
        return index
    except ValueError as e:
        print(f"  [SKIP] Index {index} ValueError: {e}")
        return index
    except KeyError as e:
        print(f"  [SKIP] Index {index} KeyError: {e}")
        return index
    except Exception as e:
        print(f"  [SKIP] Index {index} Unexpected {type(e)}: {e}")
        import traceback
        traceback.print_exc()
        return index

    # Build one graph per variant
    lst_results = [
        _build_single_graph(index, row, variant, num_negatives, idx_neg_start)
        for variant in graph_variants
    ]
    return lst_results


class TreeHopTrainDataset(Dataset):
    def __init__(
        self,
        embedding_model: str,
        dataset_name: str,
        dataset_type: str,
        *,
        negative_dataset: str | None = None,
        exclude_comparison=True,
        num_negatives: int = 4,
        graph_cache_dir: str | None = None,
        device=None,
        num_workers=1,
        mp_context=None
    ):
        super().__init__()
        self._re_clean_title = re.compile(r"\(.*\)")

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.negative_dataset = negative_dataset
        self.exclude_comparison = exclude_comparison
        self.num_negatives = num_negatives
        self.graph_cache_dir = graph_cache_dir
        self.device = device
        self.num_workers = num_workers
        self.mp_context = mp_context

        self.embedding_path = os.path.join(
            "embedding_data", embedding_model, dataset_name, f"{dataset_type}_dense.npy"
        )

        if self.graph_cache_dir is None and \
            not os.path.exists(self.embedding_path):
            raise FileNotFoundError(
                f"Non-existing embedding file {self.embedding_path} given no graph cache."
            )

        if os.path.exists(self.embedding_path):
            self._init_dataset_idx()
        else:
            print(f"Warning: embedding file {self.embedding_path} not found. ")

        self.df_dataset = self.load_dataset(dataset_name, dataset_type)

        if self.graph_cache_dir is not None:
            self.dataset_path = os.path.join(
                self.graph_cache_dir, f"{self.dataset_name}_{self.dataset_type}_df_dataset.pkl"
            )
            self.graph_cache_path = os.path.join(
                self.graph_cache_dir, f"{self.dataset_name}_{self.dataset_type}_dgl_graph.bin"
            )

            if self.has_graph_cache():
                self.load_graph_cache()
            else:
                print("No graph cache found, creating graphs and saving to cache...")
                self._create_graphs(
                    num_negatives=self.num_negatives,
                    device=self.device,
                    num_workers=self.num_workers,
                )
                if len(self._graphs) > 0:
                    self.save_graph_cache()
                else:
                    print("Skipping cache save as no graphs were created.")
        else:
            print("No graph cache directory specified, creating graphs without caching...")
            self._create_graphs(
                num_negatives=self.num_negatives,
                device=self.device,
                num_workers=self.num_workers,
            )

    def __getstate__(self):
        # Exclude large embedding array from pickling when sent to worker processes.
        # Workers use _n_total_contexts (an int) instead of len(self.ary_contexts).
        state = self.__dict__.copy()
        state.pop('ary_contexts', None)
        return state

    def _init_dataset_idx(self):
        ary_main = np.load(self.embedding_path, mmap_mode="r")
        if self.negative_dataset is not None:
            # negative indices will start from len(ary_main)
            self._idx_neg_start = len(ary_main)
            ary_neg_contexts = np.load(self.negative_dataset, mmap_mode="r")
            self._n_total_contexts = self._idx_neg_start + len(ary_neg_contexts)
        else:
            self._idx_neg_start = None
            self._n_total_contexts = len(ary_main)

    def _load_ary_contexts(self):
        ary_main = np.load(self.embedding_path, mmap_mode="r")
        if self.negative_dataset is not None:
            ary_neg_contexts = np.load(self.negative_dataset, mmap_mode="r")
            # np.concatenate materializes both arrays — do it once in the main process only
            self.ary_contexts = torch.from_numpy(
                np.concatenate([ary_main, ary_neg_contexts], axis=0)
            )
        else:
            self.ary_contexts = torch.from_numpy(np.array(ary_main))

    def load_dataset(self, dataset_name: str, dataset_type: str):
        if dataset_type == "train":
            df_dataset = pd.read_json(
                f"train_data/{dataset_name}_train_processed.jsonl", lines=True, orient="records"
            )
        elif dataset_type == "eval":
            df_dataset = pd.read_json(
                f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True, orient="records"
            )
        else:
            raise LookupError(f"cannot find {dataset_type} {dataset_name}")

        return df_dataset

    def _create_graphs(
        self,
        num_negatives: int = 5,
        device=None,
        num_workers=10,
        max_pending_tasks=1000,
    ):
        """Create DGL graphs with controlled task submission.

        Args:
            df_dataset: DataFrame containing dataset records
            num_negatives: Number of negative samples per positive
            device: Device to place graphs on
            num_workers: Number of worker threads
            max_pending_tasks: Maximum number of tasks to keep in flight.
                Controls memory usage by limiting concurrent pending tasks.
        """
        lst_skip_index = []
        results = []

        # Use bounded process pool with module-level function (avoids pickling self)
        # Prepare items with all needed parameters for the module-level function
        mp_items = [
            (index, (
                index,
                row,
                num_negatives,
                self.exclude_comparison,
                getattr(self, '_idx_neg_start', None),
                self._n_total_contexts
            ))
            for index, row in self.df_dataset.iterrows()
        ]
        task_results = bounded_process_pool_map(
            _create_graph_helper,  # Use module-level function directly
            mp_items,
            num_workers=num_workers,
            max_pending=max_pending_tasks,
            progress_desc="Creating graphs",
            mp_context=self.mp_context
        )

        # Load embeddings once in the main process.
        # __getstate__ excludes ary_contexts from the per-task pickle so workers
        # never receive (or load) the large tensor — only idx_neg_start and
        # _n_total_contexts (plain ints) travel with the pickled self.
        self._load_ary_contexts()

        # Process results after all tasks complete
        for task_result in task_results:
            if not task_result.is_success:
                # Exception occurred
                print(f"  [ERROR] Index {task_result.index} raised"
                      f" {type(task_result.result)}: {task_result.result}")
                lst_skip_index.append(task_result.index)
            elif isinstance(task_result.result, int):
                # Helper returned index to indicate skip
                lst_skip_index.append(task_result.result)
            else:
                # _create_graph_helper returns either a single tuple (most queries)
                # or a list of tuples (comparison queries with ordering variants).
                raw = task_result.result
                graph_data_list = raw if isinstance(raw, list) else [raw]

                for graph_data in graph_data_list:
                    lst_node_out, lst_node_in, lst_y, num_nodes, lst_idx_ctxs = graph_data
                    # Build DGL graph and torch tensors in the main process.
                    # DGL/torch objects must not be created inside subprocess workers.
                    graph = dgl.graph(
                        (lst_node_out, lst_node_in),
                        idtype=torch.int32,
                        num_nodes=num_nodes,
                    )
                    graph.ndata["y"] = torch.as_tensor(lst_y, dtype=torch.float32)
                    # Fill embedding data from the main process's ary_contexts.
                    # lst_idx_ctxs[0] is always the query (dataframe row) index.

                    graph.ndata["rep"] = torch.from_numpy(self.ary_contexts[lst_idx_ctxs]).float()
                    graph.ndata["h"] = torch.from_numpy(
                        self.ary_contexts[lst_idx_ctxs[0]].repeat(graph.num_nodes(), 1)
                        ).float()
                    results.append((task_result.index, graph))

        self._graphs = [res[1] for res in sorted(results, key=lambda x: x[0])]
        self.df_dataset.drop(index=lst_skip_index, inplace=True)
        print(f"Skipped {len(lst_skip_index)} records. Total trainable: {len(self.df_dataset)}")

        if device is not None:
            self.to(device)

        return self._graphs

    @property
    def graphs(self):
        if hasattr(self, "_graphs"):
            return self._graphs

        return self._create_graphs(
            num_negatives=self.num_negatives,
            device=self.device,
            num_workers=self.num_workers,
        )

    def save_graph_cache(self):
        if not isinstance(self.graph_cache_dir, str):
            raise ValueError(
                f"graph_cache_dir must be a string to save graph cache, "
                f"but got {type(self.graph_cache_dir)}"
            )
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        # save graphs and labels
        dgl.save_graphs(self.graph_cache_path, self.graphs)

        # save dataset and other information
        self.df_dataset.to_pickle(self.dataset_path)

    def load_graph_cache(self):
        # load processed data from directory `graph_cache_path`
        print(f"Loading graph cache '{self.graph_cache_path}'")
        self._graphs, _ = dgl.load_graphs(self.graph_cache_path)
        for g in self._graphs:
            if g.ndata["y"].dtype != torch.float:
                g.ndata["y"] = g.ndata["y"].float()
            if g.ndata["h"].dtype != torch.float:
                g.ndata["h"] = g.ndata["h"].float()
            if g.ndata["rep"].dtype != torch.float:
                g.ndata["rep"] = g.ndata["rep"].float()

        # load dataset and other information
        print(f"Loading dataset '{self.dataset_path}'")
        self.df_dataset = pd.read_pickle(self.dataset_path)

        if self.device is not None:
            self.to(self.device)

    def has_graph_cache(self):
        return os.path.exists(self.graph_cache_path) \
            and os.path.exists(self.dataset_path)

    @classmethod
    def is_acyclic(cls, graph: Union[dgl.DGLGraph, nx.Graph]):
        if isinstance(graph, dgl.DGLGraph):
            nx_graph = dgl.to_networkx(graph)

        try:
            return nx.is_directed_acyclic_graph(nx_graph)
        except nx.NetworkXUnfeasible:
            return False

    def reset(self):
        for graph in self._graphs:
            # h_0 = rep_q
            graph.ndata["h"] = graph.ndata["rep"][0].detach().repeat(graph.num_nodes(), 1)
            if "sim" in graph.edata:
                del graph.edata["sim"]

    def to(self, device):
        if not hasattr(self, "_graphs"):
            raise AttributeError(
                "Graphs have not been created yet."
                "Call graphs property or _create_graphs() first."
            )

        if self.device == device:
            return  # Already on the target device

        self._graphs = [graph.to(device) for graph in self._graphs]
        self.device = device

    def __getitem__(self, index: int) -> tuple:
        return self._graphs[index]
    
    def __len__(self):
        return len(self._graphs)


class TreeHopInferenceDataset(Dataset):
    def __init__(
        self,
        graphs
    ):
        super().__init__()
        self.graphs = graphs

    def __getitem__(self, index: int) -> tuple:
        return self.graphs[index]
    
    def __len__(self):
        return len(self.graphs)
