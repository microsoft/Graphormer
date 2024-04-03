import torch

import fairseq.utils as fairseq_utils


def graphormer_default_add_args(parser):
    """Add model-specific arguments to the parser."""
    # Arguments related to dropout
    parser.add_argument(
        "--dropout", type=float, metavar="D", help="dropout probability"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for" " attention weights",
    )
    parser.add_argument(
        "--act-dropout",
        type=float,
        metavar="D",
        help="dropout probability after" " activation in FFN",
    )

    # Arguments related to hidden states and self-attention
    parser.add_argument(
        "--encoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--encoder-layers", type=int, metavar="N", help="num encoder layers"
    )
    parser.add_argument(
        "--encoder-attention-heads",
        type=int,
        metavar="N",
        help="num encoder attention heads",
    )

    # Arguments related to input and output embeddings
    parser.add_argument(
        "--encoder-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dimension",
    )
    parser.add_argument(
        "--share-encoder-input-output-embed",
        action="store_true",
        help="share encoder input" " and output embeddings",
    )
    parser.add_argument(
        "--encoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the encoder",
    )
    parser.add_argument(
        "--no-token-positional-embeddings",
        action="store_true",
        help="if set, disables positional embeddings" " (outside self attention)",
    )
    parser.add_argument(
        "--max-positions", type=int, help="number of positional embeddings to learn"
    )

    # Arguments related to parameter initialization
    parser.add_argument(
        "--apply-graphormer-init",
        action="store_true",
        help="use custom param initialization for Graphormer",
    )

    # Arguments related to fintuning tricks
    parser.add_argument(
        "--layer-scale",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--droppath-prob",
        type=float,
        default=0.0,
    )

    # misc params
    parser.add_argument(
        "--activation-fn",
        choices=fairseq_utils.get_available_activation_fns(),
        help="activation function to use",
    )
    parser.add_argument(
        "--encoder-normalize-before",
        action="store_true",
        help="apply layernorm before each encoder block",
    )
    parser.add_argument(
        "--sandwich-norm",
        default=False,
        action="store_true",
        help="use sandwich layernorm for the encoder block",
    )


def guess_fisrt_load(named_parameters, state_dict, name):
    first_load = False
    # guess is this the first time we are loading the checkpoint
    if set(named_parameters.keys()) != set(state_dict.keys()):
        first_load = True

    if not first_load:
        for k in named_parameters:
            if named_parameters[k].shape != state_dict[k].shape:
                first_load = True
                break

    return first_load


def guess_load_from_pm6(named_parameters, state_dict, name):
    upgrade_from_pm6 = False
    if any("final_sandwich_layer_norm" in k for k in state_dict.keys()):
        upgrade_from_pm6 = True
    return upgrade_from_pm6


def upgrade_state_dict_named_from_pretrained(named_parameters, state_dict, name):
    from_pm6 = guess_load_from_pm6(named_parameters, state_dict, name)
    if from_pm6:
        upgrade_state_dict_named_from_pm6_ckpt(named_parameters, state_dict, name)
        msg = "Upgraded state_dict from pm6 checkpoint"
    else:
        upgrade_state_dict_named_from_m3_ckpt(named_parameters, state_dict, name)
        msg = "Upgraded state_dict from m3 checkpoint"
    return msg


def upgrade_state_dict_named_from_pm6_ckpt(named_parameters, state_dict, name):
    def upgrade_pm6_keys(key_name):
        new_key_name = key_name.replace("sentence_encoder", "graph_encoder")
        new_key_name = new_key_name.replace(
            "self_attn_sandwich_layer_norm", "self_attn_layer_norm_sandwich"
        )
        new_key_name = new_key_name.replace(
            "final_layer_norm", "final_layer_norm_sandwich"
        )
        new_key_name = new_key_name.replace(
            "final_sandwich_layer_norm", "final_layer_norm"
        )
        return new_key_name

    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = upgrade_pm6_keys(key)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)

    zero_init_keys = ["role_encoder", "pos_encoder"]

    for key in named_parameters:
        if any(x in key for x in zero_init_keys):
            state_dict[key] = torch.zeros_like(named_parameters[key].data)

    to_remove_keys = [
        "masked_lm_pooler",
        "regression_lm_head_list",
        "regression_embed_out_list",
        "regression_ln_list",
    ]
    _tmp = []
    for key in state_dict.keys():
        if any(x in key for x in to_remove_keys):
            _tmp.append(key)
    for key in _tmp:
        state_dict.pop(key)


def upgrade_state_dict_named_from_m3_ckpt(named_parameters, state_dict, name):
    to_remove_keys = [
        "encoder.embed_outs",
        "encoder.edge_out",
        "encoder.spatial_out",
    ]
    _tmp = []
    for key in state_dict.keys():
        if any(x in key for x in to_remove_keys):
            _tmp.append(key)
    for key in _tmp:
        state_dict.pop(key)
