import m3gnet
from m3gnet.models import M3GNet, Potential
from pymatgen.core import Structure, Lattice
from PyAstronomy import pyasl
import torch
import numpy as np
import tensorflow as tf

import gc
import os 
 
dir_path = os.path.dirname(os.path.realpath(__file__))

tf.config.set_visible_devices([], 'GPU')

m3gnet_model = M3GNet.from_dir(f"{dir_path}/matbench_bandgap")
m3gnet_pot = Potential(m3gnet_model)

pyasl_an = pyasl.AtomicNo()

def atomic_numbers_to_element_name(atomic_numbers):
    return [pyasl_an.getElSymbol(int(atomic_number)) for atomic_number in atomic_numbers]

def get_bandgap_and_derivative(num_atoms, lattice, pos, m3gnet_pot):
    ic("before construct structure")
    structure = Structure(Lattice(lattice.cpu().numpy()), ["C" for _ in range(num_atoms)], pos.cpu().numpy())
    ic("after construct structure")
    res = m3gnet_pot.get_efs(structure)
    gc.collect()
    return res

def get_bandgap_and_derivative_batched(batched_data, t, bandgap_class, center_pos):
    bandgaps = []
    pos_derivatives = []
    lattice_derivatives = []
    device = batched_data["pos"].device
    n_tokens = batched_data["pos"].size()[1]
    for sid, lattice, natoms, pos in zip(batched_data["sid"], batched_data["cell_pred"], batched_data["natoms"], batched_data["pos"] + center_pos):
        bandgap, pos_derivative, lattice_derivative = get_bandgap_and_derivative(natoms - 8, lattice, pos[:natoms - 8], m3gnet_pot)
        factor = float(-1.0 / (1.0 + torch.exp(2.0 - torch.tensor(bandgap.numpy())))) if bandgap_class == -1 else \
                 float(1.0 / (1.0 + torch.exp(torch.tensor(bandgap.numpy()) - 2.0)))
        bandgaps.append(bandgap.numpy())
        vol = torch.linalg.det(lattice.cpu())
        lattice_derivative *= vol
        lattice_derivative = torch.matmul(torch.linalg.inv(lattice.cpu()), torch.tensor(lattice_derivative.numpy())) * factor
        pos_derivative_padded = np.zeros([n_tokens, 3], dtype=np.float)
        pos_derivative_padded[:natoms - 8] = pos_derivative.numpy() * factor
        pos_derivatives.append(pos_derivative_padded)
        lattice_derivatives.append(lattice_derivative.numpy()[0])
    bandgaps = torch.tensor(np.array(bandgaps), device=device)
    pos_derivatives = torch.tensor(np.array(pos_derivatives), device=device)
    grad_norm_clip = 1.0
    pos_derivatives = torch.min(torch.max(pos_derivatives, -grad_norm_clip * torch.ones_like(pos_derivatives)), grad_norm_clip * torch.ones_like(pos_derivatives))
    lattice_derivatives = -torch.tensor(np.array(lattice_derivatives), device=device) / 160.21766208
    lattice_derivatives = torch.min(torch.max(lattice_derivatives, -grad_norm_clip * torch.ones_like(lattice_derivatives)), \
                                    grad_norm_clip * torch.ones_like(lattice_derivatives))
    lattice_derivatives = torch.zeros_like(lattice_derivatives)
    if t % 100 == 0:
        tf.keras.backend.clear_session()
    return bandgaps, pos_derivatives, lattice_derivatives

def get_bandgap_and_derivative_batched_multi_intervals(batched_data, t, center_pos, interval_index, target_bandgap_softmax_temperature):
    bandgaps = []
    pos_derivatives = []
    lattice_derivatives = []
    device = batched_data["pos"].device
    dtype = batched_data["pos"].dtype
    intervals = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)
    n_tokens = batched_data["pos"].size()[1]
    for sid, lattice, natoms, pos in zip(batched_data["sid"], batched_data["cell_pred"], batched_data["natoms"], batched_data["pos"] + center_pos):
        bandgap, pos_derivative, lattice_derivative = get_bandgap_and_derivative(natoms - 8, lattice, pos[:natoms - 8], m3gnet_pot)
        softmax_scores = -torch.abs(torch.tensor(float(bandgap.numpy())) - intervals) * target_bandgap_softmax_temperature
        max_score = torch.max(softmax_scores)
        softmax_scores -= max_score
        softmax_probs = torch.exp(softmax_scores)
        softmax_norm = torch.sum(softmax_probs)
        softmax_probs_remain = softmax_norm - softmax_probs[interval_index]
        factor = float(softmax_probs_remain / softmax_norm)
        bandgap_val = float(bandgap.numpy())
        if bandgap_val > intervals[interval_index]:
            factor *= -target_bandgap_softmax_temperature
        else:
            factor *= target_bandgap_softmax_temperature
        bandgaps.append(bandgap.numpy())
        vol = torch.linalg.det(lattice.cpu())
        lattice_derivative *= vol
        lattice_derivative = torch.matmul(torch.linalg.inv(lattice.cpu()), torch.tensor(lattice_derivative.numpy())) * factor
        pos_derivative_padded = np.zeros([n_tokens, 3], dtype=np.float)
        pos_derivative_padded[:natoms - 8] = pos_derivative.numpy() * factor
        pos_derivatives.append(pos_derivative_padded)
        lattice_derivatives.append(lattice_derivative.numpy()[0])
    bandgaps = torch.tensor(np.array(bandgaps), device=device)
    pos_derivatives = torch.tensor(np.array(pos_derivatives), device=device)
    grad_norm_clip = 1.0
    pos_derivatives = torch.min(torch.max(pos_derivatives, -grad_norm_clip * torch.ones_like(pos_derivatives)), grad_norm_clip * torch.ones_like(pos_derivatives))
    lattice_derivatives = -torch.tensor(np.array(lattice_derivatives), device=device) / 160.21766208
    lattice_derivatives = torch.min(torch.max(lattice_derivatives, -grad_norm_clip * torch.ones_like(lattice_derivatives)), \
                                    grad_norm_clip * torch.ones_like(lattice_derivatives))
    # skip lattice grad
    lattice_derivatives = torch.zeros_like(lattice_derivatives)
    if t % 100 == 0:
        tf.keras.backend.clear_session()
    return bandgaps, pos_derivatives, lattice_derivatives
