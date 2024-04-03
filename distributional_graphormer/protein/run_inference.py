#!/usr/bin/env python
import os
import pickle
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from common import config as cfg
from model import geometry, so3
from model.main_model import MainModel as model_fn
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def xyz2pdb(seq, CA, N, C):
    one_to_three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
        "X": "UNK",
    }
    line = "ATOM%7i  %s  %s A%4i    %8.3f%8.3f%8.3f  1.00  0.00           C"
    ret = []
    for i in range(CA.shape[0]):
        ret.append(
            line
            % (
                3 * i + 1,
                "CA",
                one_to_three[seq[i]],
                i + 1,
                CA[i][0],
                CA[i][1],
                CA[i][2],
            )
        )
        ret.append(
            line
            % (3 * i + 2, " C", one_to_three[seq[i]], i + 1, C[i][0], C[i][1], C[i][2])
        )
        ret.append(
            line
            % (3 * i + 3, " N", one_to_three[seq[i]], i + 1, N[i][0], N[i][1], N[i][2])
        )
    ret.append("TER")
    return ret


def get_checkpoint_path(step):
    if step.isdigit():
        step = int(step)
        model_path = os.path.join(cfg.model_dir, "checkpoint-step-%s.pth" % step)
    else:
        model_path = step
    return model_path


def load_model(step):
    model = model_fn()
    checkpoint_path = get_checkpoint_path(step)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model


def _inference_fn(
    model,
    single_repr,
    pair_repr,
    tr_init,
    rot_mat_init,
    save_full_state=False,
    use_tqdm=True,
):
    device = single_repr.device

    inference_steps = model.n_time_step

    def get_t_schedule(inference_steps):
        return np.linspace(1, 0, inference_steps + 1)[:-1]

    t_schedule = get_t_schedule(inference_steps=inference_steps)
    tr_schedule, rot_schedule = t_schedule, t_schedule

    tr_sigma_min, tr_sigma_max = model.tr_sigma_min, model.tr_sigma_max
    rot_sigma_min, rot_sigma_max = model.rot_sigma_min, model.rot_sigma_max

    def t_to_sigma(t_tr, t_rot):
        T_sigma = (tr_sigma_min ** (1 - t_tr)) * (tr_sigma_max ** (t_tr))
        IR_sigma = (rot_sigma_min ** (1 - t_rot)) * (rot_sigma_max ** (t_rot))
        return T_sigma, IR_sigma

    def init_conformer(feature):
        L = feature.shape[0]
        random_tr = torch.zeros(L, 3).normal_(mean=0, std=tr_sigma_max)
        torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
        random_rot = torch.from_numpy(R.random(num=L).as_matrix()).float()
        return random_tr, random_rot

    if tr_init is None or rot_mat_init is None:
        tr, rot_mat = init_conformer(single_repr)
        tr_init, rot_mat_init = tr.clone(), rot_mat.clone()
    else:
        tr, rot_mat = tr_init.clone(), rot_mat_init.clone()
    tr, rot_mat = tr.to(device), rot_mat.to(device)

    if save_full_state:
        tr_list = []
        rot_mat_list = []
        tr_list.append(tr_init.clone().cpu())
        rot_mat_list.append(rot_mat_init.clone().cpu())

    # Sampling, t: 1 -> 0
    start_time = time.time()

    for t_idx in tqdm(range(inference_steps), disable=not use_tqdm):
        t_tr, t_rot = tr_schedule[t_idx], rot_schedule[t_idx]
        dt_tr = (
            tr_schedule[t_idx] - tr_schedule[t_idx + 1]
            if t_idx < inference_steps - 1
            else tr_schedule[t_idx]
        )
        dt_rot = (
            rot_schedule[t_idx] - rot_schedule[t_idx + 1]
            if t_idx < inference_steps - 1
            else rot_schedule[t_idx]
        )

        tr_sigma, rot_sigma = t_to_sigma(t_tr, t_rot)

        # predict update
        with torch.no_grad():
            tr_score, rot_score = model.forward_step(
                (tr[None], rot_mat[None]),
                torch.zeros((1, tr.shape[0]), dtype=bool, device=tr.device),
                torch.tensor([t_idx]).to(device),
                single_repr[None],
                pair_repr[None],
            )
            tr_score, rot_score = tr_score[0], rot_score[0]
            tr_score /= tr_sigma
            rot_score *= so3.score_norm(torch.tensor([rot_sigma]))[0]
        # tr_score: (L, 3), rot_score: (L, 3)

        tr_g = tr_sigma * torch.sqrt(
            torch.tensor(2 * np.log(tr_sigma_max / tr_sigma_min))
        )
        rot_g = (
            2
            * rot_sigma
            * torch.sqrt(torch.tensor(np.log(rot_sigma_max / rot_sigma_min)))
        )

        tr_perturb_nr = tr_g**2 * dt_tr * tr_score
        rot_perturb_nr = rot_g**2 * dt_rot * rot_score

        tr_perturb = tr_perturb_nr
        rot_perturb = rot_perturb_nr

        rot_mat_perturb_nr = geometry.axis_angle_to_matrix(rot_perturb_nr)
        rot_mat_perturb = geometry.axis_angle_to_matrix(rot_perturb)

        # update conformer
        tr_mean = tr + tr_perturb_nr
        rot_mat_mean = torch.bmm(rot_mat_perturb_nr, rot_mat)

        if save_full_state:
            tr_list.append(tr_mean.clone().cpu())
            rot_mat_list.append(rot_mat_mean.clone().cpu())

        tr = tr + tr_perturb
        rot_mat = torch.matmul(rot_mat_perturb, rot_mat)

    x = torch.norm(tr_mean[1:] - tr_mean[:-1], dim=-1)
    print(
        f"CA-CA distance: {x.mean():.3f} +- {x.std():.3f} max: {x.max():.3f} min: {x.min():.3f}, len: {tr.shape[0]}, time: {time.time() - start_time:.3f}"
    )

    if not save_full_state:
        return tr_init, rot_mat_init, tr_mean, rot_mat_mean
    else:
        return tr_list, rot_mat_list


def convert_to_CANC(tr, rot_mat):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    CA = tr
    N_ref = torch.tensor([1.45597958, 0.0, 0.0])
    C_ref = torch.tensor([-0.533655602, 1.42752619, 0.0])
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref) + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref) + CA
    return CA, N, C


def write_to_pdb(seq, tr, rot_mat, file):
    CA, N, C = convert_to_CANC(tr, rot_mat)
    with open(file, "w") as fp:
        lines = xyz2pdb(seq, CA, N, C)
        fp.write("\n".join(lines))


def write_to_npz(tr, rot_mat, file):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    data = {
        "tr": tr.numpy(),
        "rot_mat": rot_mat.numpy(),
    }
    np.savez(file, **data)


@click.command()
@click.option(
    "-c", "--checkpoint", help="Step to evaluate (e.g. 100000)", required=True
)
@click.option("-i", "--pkl", required=True)
@click.option("-s", "--fasta", required=True)
@click.option("-o", "--output", required=True)
@click.option("-n", "--num-samples", default=1)
@click.option("-p", "--output-prefix", default="")
@click.option("--init-state", required=False)
@click.option("--save-full-state/--no-save-full-state", default=False)
@click.option("--use-tqdm/--no-use-tqdm", default=False)
@click.option("--use-gpu/--no-use-gpu", default=False)
def inference(
    checkpoint,
    pkl,
    fasta,
    output,
    output_prefix,
    num_samples,
    init_state,
    save_full_state,
    use_tqdm,
    use_gpu,
):
    model = load_model(checkpoint)
    model = model.eval()

    if pkl.endswith(".list"):
        pkl_list = open(pkl, "r").readlines()
        fasta_list = open(fasta, "r").readlines()
        output_list = open(output, "r").readlines()
        pkl_list = [pkl.strip() for pkl in pkl_list]
        fasta_list = [fasta.strip() for fasta in fasta_list]
        output_list = [output.strip() for output in output_list]
        assert len(pkl_list) == len(fasta_list) == len(output_list)
    else:
        pkl_list = [pkl]
        fasta_list = [fasta]
        output_list = [output]

    for pkl, fasta, output in zip(pkl_list, fasta_list, output_list):
        try:
            pkl_data = pickle.load(open(pkl, "rb"))
            if "representations" in pkl_data:
                pkl_data = pkl_data["representations"]
            single_repr = torch.from_numpy(pkl_data["single"]).float()
            pair_repr = torch.from_numpy(pkl_data["pair"]).float()
            seq = open(fasta, "r").readlines()[1].strip()
            assert len(seq) == single_repr.shape[0]

            if use_gpu and torch.cuda.is_available():
                model = model.cuda()
                single_repr = single_repr.cuda()
                pair_repr = pair_repr.cuda()

            if init_state is not None:
                init_data = np.load(init_state)
                tr_init = torch.from_numpy(init_data["tr"]).float()
                rot_mat_init = torch.from_numpy(init_data["rot_mat"]).float()
            else:
                tr_init = None
                rot_mat_init = None

            for i in range(num_samples):
                ofilename = output_prefix + f"{output}_{i}.pdb"
                ofilename_init = output_prefix + f"{output}_{i}_init_state.npz"
                ofilename_final = output_prefix + f"{output}_{i}_final_state.npz"

                if os.path.exists(ofilename) and os.path.exists(ofilename_init):
                    print(
                        f"Skipping {i + 1}/{num_samples} samples, {ofilename} already exists"
                    )
                    continue

                if not save_full_state:
                    tr_init_ret, rot_mat_init_ret, tr, rot_mat = _inference_fn(
                        model,
                        single_repr,
                        pair_repr,
                        tr_init,
                        rot_mat_init,
                        save_full_state,
                        use_tqdm,
                    )
                else:
                    tr_list, rot_mat_list = _inference_fn(
                        model,
                        single_repr,
                        pair_repr,
                        tr_init,
                        rot_mat_init,
                        save_full_state,
                        use_tqdm,
                    )
                    tr_init_ret, rot_mat_init_ret, tr, rot_mat = (
                        tr_list[0],
                        rot_mat_list[0],
                        tr_list[-1],
                        rot_mat_list[-1],
                    )
                print(
                    f"Finished {i + 1}/{num_samples} samples, writing to {ofilename} and {ofilename_init}"
                )

                if not save_full_state:
                    write_to_pdb(seq, tr, rot_mat, ofilename)
                else:
                    with open(ofilename, "w") as fp:
                        for idx, tr_rot_mat in enumerate(zip(tr_list, rot_mat_list)):
                            tr, rot_mat = tr_rot_mat
                            CA, N, C = convert_to_CANC(tr, rot_mat)
                            lines = xyz2pdb(seq, CA, N, C)
                            prefix = f"MODEL        {idx}\n"
                            fp.write(prefix)
                            fp.write("\n".join(lines))
                            fp.write("\nENDMDL\n")

                write_to_npz(tr_init_ret, rot_mat_init_ret, ofilename_init)
                write_to_npz(tr, rot_mat, ofilename_final)
        except Exception as e:
            print(f"Error processing {pkl}, {fasta}, {output}")
            print(str(e))


if __name__ == "__main__":
    inference()
