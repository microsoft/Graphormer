import torch


def sample_time_step(n, num_timesteps, device):
    time_step = torch.randint(0, num_timesteps, size=(n // 2 + 1,), device=device)
    time_step = torch.cat([time_step, num_timesteps - time_step - 1], dim=0)[:n]
    return time_step
