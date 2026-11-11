from uuid import uuid4
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Iterable, Union, Dict
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from utils import NodeType


class TreeHopGraph(nx.DiGraph):
    def __init__(
        self,
        query: str,
        passages: List[List[dict]],
        top_n: int,
        min_ranking: int | None = None,
        score: Iterable | None = None,
        query_sim: Iterable | None = None,
        gen_sim: Iterable | None = None,
        prune_redundant=True,
        prune_layer_top: Union[int, bool] = True,
        retriever_ids: NDArray | None = None,
    ):
        super().__init__()
        self.top_n = top_n
        self.layerwise_top_pruning = top_n if prune_layer_top is True else prune_layer_top

        self._map_uuid: Dict[Union[int, str], int] \
            = {NodeType.query.value: NodeType.query.value}
        self._unused_dummy = -1
        self._query_passage_mask: NDArray = np.ones((1, 1), dtype=bool)
        self._last_passage_layer_ids = None
        self._previous_query_ids = None

        if retriever_ids is None:
            self._passage_ids = {0: {self._unused_dummy}}
        else:
            self._passage_ids = {i: {self._unused_dummy}
                                 for i in np.unique(retriever_ids)}

        self.add_node(NodeType.query.value, title=f'"{query}"', text='', mask=True)
        self.add_passage_layer(passages, score=score,
                               query_sim=query_sim, gen_sim=gen_sim,
                               min_ranking=min_ranking,
                               prune_redundant=prune_redundant,
                               prune_layer_top=self.layerwise_top_pruning,
                               retriever_ids=retriever_ids)

    def _id_to_uuid(self, id_):
        uuid = str(uuid4())
        self._map_uuid[uuid] = id_
        return uuid

    def _uuid_to_id(self, uuid):
        return self._map_uuid[uuid]

    def _add_passage(self, psg, **kwargs):
        psg_uuid = self._id_to_uuid(psg["id"])
        self.add_node(
            psg_uuid,
            id=psg["id"],
            title=f'"{psg["title"]}"',
            text=f'"{psg["text"]}"',
            **kwargs
        )
        return psg_uuid

    def _mask_passage_layer(
        self,
        df_passage_layer: pd.DataFrame,
        num_last_queries: int,
        scoring_col: str = "score",
        prune_redundant: bool = True,
        prune_layer_top: Union[int, bool] = True,
        min_ranking: int | None = None,
        # query_sim: Iterable = None,
        # gen_sim: Iterable = None,
        eps: float = 1e-5
    ) -> NDArray:
        if prune_redundant:
            srs_duplicate_mask = (
                df_passage_layer
                .apply(lambda df: df["id"] in self._passage_ids[df["retriever_idx"]],
                       axis=1)
            )
        else:
            srs_duplicate_mask = pd.Series([False] * df_passage_layer.shape[0])

        grouped_layer = df_passage_layer.groupby(["query_idx", "retriever_idx"], sort=False)
        srs_max_score = grouped_layer[scoring_col].transform("max")
        srs_min_score = grouped_layer[scoring_col].transform("min")
        # srs_mean_score = grouped_layer.transform("mean")
        # srs_std_score = grouped_layer.transform("std")
        # z-scores on similarities
        # df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = \
        #     (df_passage_layer[scoring_col] - srs_mean_score) / srs_std_score
        # range of cosine similarity is [-1, 1], scale to [0, 1]
        # df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = (df_passage_layer[f"{scoring_col}"] + 1.) / 2.
        # df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = (df_passage_layer[f"{scoring_col}"] - df_passage_layer[f"{scoring_col}"].min()) \
        #     / (df_passage_layer[f"{scoring_col}"].max() - df_passage_layer[f"{scoring_col}"].min() + eps)
        df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = \
            (df_passage_layer[f"{scoring_col}"] - srs_min_score) \
            / (srs_max_score - srs_min_score + eps)
        # df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = np.exp(df_passage_layer[f"{scoring_col}"])

        df_passage_layer.fillna({f"scaled_z__{scoring_col}": eps}, inplace=True)
        # df_passage_layer[f"std_z__{scoring_col}"] = df_passage_layer[f"z__{scoring_col}"].std()

        # # re-normalize after temperature scaling
        # srs_sum_scaled_z = grouped_layer[f"scaled_z__{scoring_col}"].transform("sum")
        # df_passage_layer.loc[:, f"scaled_z__{scoring_col}"] = \
        #     df_passage_layer[f"scaled_z__{scoring_col}"] / (srs_sum_scaled_z + eps)

        # # exclude duplicated retrieved passages except for those who score highest
        # df[f"max_{scoring_col}"] = df.groupby("id")[scoring_col].transform("max")
        # srs_highest_score_mask = df[f"max_{scoring_col}"] == df[scoring_col]

        # srs_duplicate_mask = srs_duplicate_mask & ~srs_highest_score_mask
        df_passage_layer.loc[srs_duplicate_mask, f"scaled_z__{scoring_col}"] = self._unused_dummy

        srs_score_mask = pd.Series(False, index=df_passage_layer.index)

        idx_rank = np.argsort(
            df_passage_layer[f"scaled_z__{scoring_col}"].to_numpy()
        )

        if prune_layer_top:
            idx_rank = np.argsort(idx_rank)
            # gives an array where
            # each element's value is its rank (0-indexed) within the original array
            # idx_rank = np.argsort(idx_rank)
            # idx_rank[srs_duplicate_mask] = self._unused_dummy

            if min_ranking is None:
                min_ranking = max(idx_rank.max(axis=None) - prune_layer_top + 1, 0)

            srs_score_mask |= (idx_rank < min_ranking)
        else:
            ary_bool_ranking = np.zeros(len(df_passage_layer), dtype=bool)
            # TODO: might try deduplicate when min-max scaling
            if min_ranking is not None:
                ary_bool_ranking[idx_rank[-min_ranking:]] = True

            srs_score_mask[~srs_duplicate_mask & ary_bool_ranking] = False

        ary_mask = (srs_score_mask | srs_duplicate_mask).to_numpy()
        # mask out low-ranking passages
        # reshape to (num_last_queries, top_n)
        out = ~ary_mask.reshape(num_last_queries, -1)
        return out

    def _register_retriever(self, retriever_ids):
        for retriever_idx in np.unique(retriever_ids):
            if retriever_idx not in self._passage_ids:
                self._passage_ids[retriever_idx] = {self._unused_dummy}

    def _is_passage_retrieved(self, id_: str) -> bool:
        return any(
            id_ in retriever_ids
            for retriever_ids in self._passage_ids.values()
        )

    def add_passage_layer(
        self,
        passage_layer,
        prune_redundant=True,
        prune_layer_top: Union[int, bool] = True,
        *,
        top_n: int | None = None,
        min_ranking: int | None = None,
        score: Iterable | None = None,
        query_sim: Iterable | None = None,
        gen_sim: Iterable | None = None,
        retriever_ids: NDArray | None = None,
    ) -> None:
        last_query_ids = self.get_last_query_ids()
        assert len(last_query_ids) == len(passage_layer), \
            f"number of new passages must match with number of last query nodes, " \
            f"got {len(last_query_ids)} and {len(passage_layer)}"
            
        if len(passage_layer) == 0:
            self._filtered_passages = []
            self._query_passage_mask = np.zeros((0, 0), dtype=bool)
            self._last_passage_layer_ids = []
            self._previous_query_ids = last_query_ids
            return

        if top_n is not None:
            self.top_n = top_n

        if prune_layer_top is True:
            prune_layer_top = self.top_n

        n_last_q = len(passage_layer)
        df_psg_layer: pd.DataFrame = pd.DataFrame([psg for passages in passage_layer for psg in passages])

        lst_score_attr = ["score"]
        if score is not None:
            df_psg_layer["score"] = score
        if query_sim is not None:
            df_psg_layer["query_sim"] = query_sim
            lst_score_attr.append("query_sim")
        if gen_sim is not None:
            df_psg_layer["gen_sim"] = gen_sim
            lst_score_attr.append("gen_sim")

        assert retriever_ids is None or len(retriever_ids) == df_psg_layer.shape[0], \
            "number of retriever ids must match number of passages"

        if retriever_ids is not None:
            self._register_retriever(retriever_ids)
        elif retriever_ids is None:
            if self.has_passage_layer():
                retriever_ids = np.repeat(
                    self.get_last_query_retriever_ids(), self.top_n
                )
            else:
                retriever_ids = np.zeros(df_psg_layer.shape[0], dtype=int)

        df_psg_layer["retriever_idx"] = retriever_ids
        if self.has_passage_layer():
            # for subsequent passage layers, we need to map query indices
            df_psg_layer["query_idx"] = [
                self._uuid_to_id(id_)
                for id_ in last_query_ids
                for _ in range(self.top_n)
            ]
        else:
            # for the first passage layer, all queries are from the root query node
            df_psg_layer["query_idx"] = 0

        self._query_passage_mask = self._mask_passage_layer(
            df_psg_layer,
            n_last_q,
            prune_redundant=prune_redundant,
            prune_layer_top=prune_layer_top,
            min_ranking=min_ranking,
            # query_sim=query_sim,
            # gen_sim=gen_sim,
        )

        i_psg = 0
        filtered_passages = []
        last_passage_layer_ids = []
        iter_retriever_ids = iter(retriever_ids)
        for passages, masks, idx_query in zip(
            passage_layer, self._query_passage_mask,
            last_query_ids,
            strict=True
        ):
            lst_passage_ids = []
            for i_psg, psg in enumerate(passages):
                retr_idx = next(iter_retriever_ids)
                if prune_redundant \
                    and psg["id"] in self._passage_ids[retr_idx]:
                    # apply redundant pruning
                    masks[i_psg] = False

                # record filtered passages for later use
                if masks[i_psg] \
                    and (not prune_redundant
                         or psg["id"] not in self._passage_ids[retr_idx]):
                    # the same passage could be retrieved by different retrievers
                    # thus we only add it once
                    if not prune_redundant or not self._is_passage_retrieved(psg["id"]):
                        filtered_passages.append(psg)

                    self._passage_ids[retr_idx].add(psg["id"])

                # add passage node and edge to graph
                score_attr = df_psg_layer.loc[i_psg, lst_score_attr] # type: ignore
                psg_uuid = self._add_passage(
                    psg, mask=masks[i_psg],
                    retriever_idx=self.nodes[idx_query].get("retriever_idx", 0)
                )
                self.add_edge(idx_query, psg_uuid,
                              **score_attr.to_dict())
                i_psg += 1
                lst_passage_ids.append(psg_uuid)

            # disable mask on query nodes after adding their passage layer
            self.nodes[idx_query]["mask"] = False
            last_passage_layer_ids.append(lst_passage_ids)

        # broadcast retriever ids to decendant passage nodes
        if retriever_ids is not None:
            nx.set_node_attributes(
                self,
                {i_psg: i_retriever
                 for i_psg, i_retriever
                 in zip([id_ for ids in last_passage_layer_ids for id_ in ids],
                        retriever_ids,
                        strict=True)},
                name="retriever_idx"
            )

        self._filtered_passages = filtered_passages
        self._last_passage_layer_ids = last_passage_layer_ids
        # for convenience, record previous query ids for passage layers
        # because query nodes' mask will be disabled after adding passage layer
        self._previous_query_ids = last_query_ids

    def has_passage_layer(self) -> bool:
        return self._last_passage_layer_ids is not None

    @property
    def num_retrievers(self) -> int:
        return len(self._passage_ids)

    @property
    def query_passage_mask(self) -> NDArray:
        return self._query_passage_mask

    @property
    def filtered_passages(self) -> List[List[dict]]:
        return self._filtered_passages

    @property
    def last_passage_layer_ids(self) -> List[int] | None:
        if self.has_passage_layer():
            return self._last_passage_layer_ids

        raise ValueError("No passage layer found in the graph.")

    @property
    def previous_query_ids(self) -> List[int] | None:
        return self._previous_query_ids

    def get_last_query_ids(self):
        return [u for u, v in self.nodes(data=True) if v["mask"]]

    def get_last_query_retriever_ids(self):
        if not self.has_passage_layer():
            raise ValueError("No passage layer found in the graph.")

        return np.asarray([
            self.nodes[id_].get("retriever_idx", 0)
            for psg_uuids in self._last_passage_layer_ids # type: ignore
            for id_ in psg_uuids
            if self.nodes[id_]["mask"]
        ])

    def get_query_passage_mask_by_retriever_id(
        self,
        retriever_id: int
    ) -> np.ndarray:

        if not self.has_passage_layer() \
            or self.previous_query_ids is None:
            raise ValueError("No passage layer found in the graph.")

        # retain only passage masks retrieved by the specified retriever
        query_passage_masks = []
        for psg_ids, masks, query_id \
            in zip(self._last_passage_layer_ids, # type: ignore
                self._query_passage_mask,
                self.previous_query_ids,
                strict=True):

            # the only case where retriever_idx is not found is the query node
            # which does not have retriever_idx attribute, we need to check
            # all passage nodes connected to it
            if "retriever_idx" not in self.nodes[query_id]:
                layer_masks = []
                for psg_id, mask in zip(psg_ids, masks):
                    if self.nodes[psg_id].get("retriever_idx", 0) == retriever_id:
                        layer_masks.append(mask)

                query_passage_masks.append(layer_masks)
            # if the query node has retriever_idx attribute
            # we can directly check for its decendants
            elif self.nodes[query_id]["retriever_idx"] == retriever_id:
                query_passage_masks.append(masks)
            # else, records will be excluded because of different retriever

        query_passage_masks = np.array(query_passage_masks, dtype=bool)
        # if not query_passage_masks.any():
        #     return np.zeros(0, dtype=bool)

        return query_passage_masks

    def plot_tree(self, label_attr="id", ax=None):
        # this plot requires pygraphviz package
        pos = nx.nx_agraph.graphviz_layout(self, prog="dot")
        labels = nx.get_node_attributes(self, label_attr)
        edge_labels = nx.get_edge_attributes(self, "score")
        for key in edge_labels.keys():
            edge_labels[key] = f"{edge_labels[key]:.3f}"

        if ax is None:
            plt.figure(figsize=(25, 10))

        nx.draw(
            self,
            pos,
            arrows=True, # type: ignore
            with_labels=True,
            labels=labels,
            node_size=5,
            node_color=[[0.5, 0.5, 0.5]],
            arrowsize=4,
            ax=ax
        )

        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)
        if ax is None:
            plt.show()
        else:
            ax.show()
    
    def draw_graph(self):
        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrows': True,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(self, **options)
