import functools
import pandas as pd


def model_file_name_to_params(name):
    lst_params = name.rstrip(".pt").split('__')[1].split('&')
    d_params = dict([param.split('=') for param in lst_params])
    if "dropout" in d_params:
        d_params["dropout"] = float(d_params["dropout"])
    if "weight_decay" in d_params:
        d_params["weight_decay"] = float(d_params["weight_decay"])
    if "lr" in d_params:
        d_params["lr"] = float(d_params["lr"])
    if "temperature" in d_params:
        d_params["temperature"] = float(d_params["temperature"])
    if "batch_size" in d_params:
        d_params["batch_size"] = int(d_params["batch_size"])
    if "n_neg" in d_params:
        d_params["n_neg"] = int(d_params["n_neg"])
    if "x_size" in d_params:
        d_params["x_size"] = int(d_params["x_size"])
    if "g_size" in d_params:
        d_params["g_size"] = int(d_params["g_size"])
    if "mlp_size" in d_params:
        d_params["mlp_size"] = int(d_params["mlp_size"])
    if "n_mlp" in d_params:
        d_params["n_mlp"] = int(d_params["n_mlp"])
    if "n_head" in d_params:
        d_params["n_head"] = int(d_params["n_head"])
    if "n_layer" in d_params:
        d_params["n_layer"] = int(d_params["n_layer"])

    return d_params


_LEGACY_GATES = ("update_gate", "forget_gate", "forget_gate2")
_LEGACY_PROJECTIONS = {
    "update_attn_scale": "update_gate",
    "forget_attn_scale": "forget_gate",
    "forget_attn_scale2": "forget_gate2",
}


def remap_pre_stacking_state_dict(state_dict):
    """Move a pre-stacking checkpoint onto the stacked gate layout.

    Before the gates became stackable each one was a bare attention hanging off
    the node, e.g. ``node.update_gate.heads.*``, followed by a node-level
    projection back to the embedding size, e.g. ``node.update_attn_scale.*``.
    The two together are the first layer of a one-deep stack, so the attention
    moves to ``node.update_gate.layers.0.*`` and the projection to
    ``node.update_gate.projections.0.*``. Any other key is left untouched.
    """
    remapped = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        if len(parts) > 2 and parts[0] == "node":
            if parts[1] in _LEGACY_GATES:
                parts = ["node", parts[1], "layers", "0"] + parts[2:]
            elif parts[1] in _LEGACY_PROJECTIONS:
                parts = ["node", _LEGACY_PROJECTIONS[parts[1]], "projections", "0"] + parts[2:]

            key = '.'.join(parts)

        remapped[key] = value

    return remapped


@functools.lru_cache()
def get_tree_hop_model(state_dict: str, model_cls, device="cpu", **hf_model_kwargs):
    if not state_dict.lstrip("./").startswith("checkpoint"):
        model = model_cls.from_pretrained(state_dict, **hf_model_kwargs)
        model.to(device).compile()
        return model

    d_params = model_file_name_to_params(state_dict)
    # checkpoints predating stacked gates record no layer count; they load as a
    # one-deep stack, which holds no norm weights whatever `norm` asks for
    is_stacked = "n_layer" in d_params
    model = model_cls(
        x_size=int(d_params["x_size"]),
        g_size=int(d_params["g_size"]),
        mlp_size=int(d_params["mlp_size"]),
        n_mlp=int(d_params["n_mlp"]),
        n_head=int(d_params["n_head"]),
        n_layer=int(d_params["n_layer"]) if is_stacked else 1,
        norm=d_params.get("norm", "rms")
    )

    import torch
    pt_state_dict = torch.load(state_dict, weights_only=True, map_location=device)
    if not is_stacked:
        pt_state_dict = remap_pre_stacking_state_dict(pt_state_dict)

    model.load_state_dict(pt_state_dict)
    model.to(device).compile()
    return model


@functools.lru_cache()
def get_dataset(dataset_name):
    df_QA = pd.read_json(f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True)
    df_QA = (df_QA[~df_QA["type"].isin([#"comparison", # 2wiki
                                        # multihop_rag
                                        #"comparison_query", "null_query", "temporal_query"
                                        ])]
             .reset_index())
    df_QA["set_evidence_title"] = df_QA["supporting_facts"].apply(
        lambda lst: set([evd[0] for evd in lst])
    )
    return df_QA
