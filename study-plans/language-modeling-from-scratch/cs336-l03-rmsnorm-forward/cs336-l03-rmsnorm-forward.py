import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    rms = torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon

    return x * g * torch.rsqrt(rms)
