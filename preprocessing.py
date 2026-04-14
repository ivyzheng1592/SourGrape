import torch
import torch.nn.functional as F

def add_noise(x, std=0.02):
    """
    Add Gaussian noise with fixed standard deviation.

    Args:
    x (torch.Tensor): input tensor (any shape)
    std (float): target standard deviation of noise

    Returns:
    torch.Tensor: perturbed tensor
    """
    noise = torch.randn_like(x) * std
    return x + noise


def augment_trajectory_variable_length(
    x,
    stretch_range=(0.8, 1.25),
    scale_range=(0.8, 1.2),
    shift_range=(-0.2, 0.2),
    p_stretch=1.0,
    p_scale=1.0,
    p_shift=1.0,
):
    """
    Augment a single trajectory of shape (L, C).

    Operations:
    - random time stretching/compression
    - random amplitude scaling
    - random vertical shifting

    Returns:
        traj: tensor of shape (new_L, C)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected input shape (L, C), got {tuple(x.shape)}")

    traj = x.clone().float()
    orig_len, num_channels = traj.shape

    # 1. Random time stretch/compression
    if torch.rand(1).item() < p_stretch:
        stretch_factor = torch.empty(1).uniform_(*stretch_range).item()
        new_len = max(2, int(round(orig_len * stretch_factor)))

        # F.interpolate expects (N, C, L)
        temp = traj.transpose(0, 1).unsqueeze(0)   # (1, C, L)
        stretched = F.interpolate(
            temp,
            size=new_len,
            mode="linear",
            align_corners=False
        )
        traj = stretched[0].transpose(0, 1)        # (new_L, C)

    # 2. Random amplitude scaling
    if torch.rand(1).item() < p_scale:
        scale_factor = torch.empty(1).uniform_(*scale_range).item()
        traj = traj * scale_factor

    # 3. Random vertical shifting
    if torch.rand(1).item() < p_shift:
        shift_value = torch.empty(1).uniform_(*shift_range).item()
        traj = traj + shift_value

    return traj
