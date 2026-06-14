import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
	return torch.min(torch.tensor(1), torch.Tensor([(t / (c / sum(counts))) ** 0.5 for c in counts]))