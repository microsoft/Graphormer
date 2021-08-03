# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import math


def flag_bounded(model_forward, perturb_shape, y, optimizer, device, criterion, m=3, step_size=1e-3, mag=1e-3, mask=None):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    if mag > 0:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(device)
        perturb = perturb * mag / math.sqrt(perturb_shape[-1])
    else:
        perturb = torch.FloatTensor(
            *perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb).view(-1)
    if mask is not None:
        out = out[mask]
    loss = criterion(out, y)
    loss /= m

    for _ in range(m-1):
        # loss.backward()
        model.manual_backward(loss)
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        if mag > 0:
            perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
            exceed_mask = (perturb_data_norm > mag).to(perturb_data)
            reweights = (mag / perturb_data_norm * exceed_mask +
                         (1-exceed_mask)).unsqueeze(-1)
            perturb_data = (perturb_data * reweights).detach()

        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb).view(-1)
        if mask is not None:
            out = out[mask]
        loss = criterion(out, y)
        loss /= m

    # loss.backward()
    model.manual_backward(loss)
    optimizer.step()

    return loss, out
