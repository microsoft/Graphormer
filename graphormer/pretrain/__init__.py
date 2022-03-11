from torch.hub import load_state_dict_from_url
import torch.distributed as dist

PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv2/checkpoint_best_pcqm4mv2.pt",
    "oc20is2re_graphormer3d_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt",
    "pcqm4mv1_graphormer_base_for_molhiv":"https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}

def load_pretrained_model(pretrained_model_name):
    if pretrained_model_name not in PRETRAINED_MODEL_URLS:
        raise ValueError("Unknown pretrained model name %s", pretrained_model_name)
    if not dist.is_initialized():
        return load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True)["model"]
    else:
        pretrained_model = load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True, file_name=f"{pretrained_model_name}_{dist.get_rank()}")["model"]
        dist.barrier()
        return pretrained_model
