import torch


@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


def make_masks(batched_data):
    if "lig_mask" not in batched_data:
        n_graphs, n_atoms = batched_data["x"].shape[:2]  # G, T
        lnode = batched_data["lnode"]  # ligand atom numbers in a batch
        batched_data["lig_mask"] = ~mask_after_k_persample(n_graphs, n_atoms, lnode)
    if "pro_mask" not in batched_data:
        lnode = batched_data["lnode"]
        pnode = batched_data["pnode"]
        lp_mask = ~mask_after_k_persample(n_graphs, n_atoms, lnode + pnode)
        p_mask = lp_mask & (~batched_data["lig_mask"])
        batched_data["pro_mask"] = p_mask


def get_center_pos(batched_data, type='protein', crystal=False):
    """
    2022-11-02: Switch to get the center of the protein
    """
    # return ligand center position, return [G, 3]
    if crystal:
        # make_masks(batched_data)
        pro_mask = batched_data["pro_mask"]
        c = (
            torch.sum(batched_data["crystal_pos"] * pro_mask[:, :, None], axis=1).unsqueeze(1)
            / batched_data["pnode"][:, None, None]
        )
        return c

    make_masks(batched_data)
    lig_mask = batched_data["lig_mask"]
    pro_mask = batched_data["pro_mask"]
    if type == 'protein':
        c = (
            torch.sum(batched_data["pos"] * pro_mask[:, :, None], axis=1).unsqueeze(1)
            / batched_data["pnode"][:, None, None]
        )
    elif type == 'ligand':
        c = (
            torch.sum(batched_data["pos"] * lig_mask[:, :, None], axis=1).unsqueeze(1)
            / batched_data["lnode"][:, None, None]
        )
    return c


def tensor_merge(cond, input, other):
    return cond * input + (~cond) * other

def tensor_merge_truncated(cond, input, other, max_len):
    ret = (~cond) * other
    ret[:, :max_len, :] = cond[:, :max_len, :] * input
    return ret