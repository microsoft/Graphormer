from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from torch.hub import load_state_dict_from_url

PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt",
    # This pretrained model is temporarily unavailable.
    "oc20is2re_graphormer3d_base": "https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt",
    "pcqm4mv1_graphormer_base_for_molhiv": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}


def _load_local_pretrained_model(pretrained_model_path):
    checkpoint_path = Path(pretrained_model_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained model path does not exist: {checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise IsADirectoryError(
            f"Pretrained model path is not a file: {checkpoint_path}"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, Mapping):
        model_state = checkpoint.get("model", checkpoint)
    else:
        model_state = checkpoint

    if (
        not isinstance(model_state, Mapping)
        or not model_state
        or not all(isinstance(key, str) for key in model_state)
    ):
        raise ValueError(
            "Local pretrained model must be a non-empty state dictionary or a "
            "checkpoint containing a 'model' state dictionary"
        )
    return model_state


def load_pretrained_model(pretrained_model_name="none", pretrained_model_path=None):
    if pretrained_model_path:
        if pretrained_model_name != "none":
            raise ValueError(
                "Set either pretrained_model_name or pretrained_model_path, not both"
            )
        return _load_local_pretrained_model(pretrained_model_path)

    if pretrained_model_name not in PRETRAINED_MODEL_URLS:
        raise ValueError(f"Unknown pretrained model name: {pretrained_model_name}")
    if dist.is_initialized():
        checkpoint = load_state_dict_from_url(
            PRETRAINED_MODEL_URLS[pretrained_model_name],
            progress=True,
            file_name=f"{pretrained_model_name}_{dist.get_rank()}",
        )
        dist.barrier()
    else:
        checkpoint = load_state_dict_from_url(
            PRETRAINED_MODEL_URLS[pretrained_model_name],
            progress=True,
        )
    return checkpoint["model"]
