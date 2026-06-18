import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
	skipgram_pairs = []
	
	n = len(token_ids)
	for i in range(n):
		clamped_window = [max(0, i-window), min(i+window, n-1)]
		r = clamped_window[-1] - clamped_window[0]
		clamped_context = token_ids[clamped_window[0]: clamped_window[-1]+1].tolist()
		
		center_index = i - clamped_window[0]
		
		for j in range(len(clamped_context)):
			if center_index == j:
				pass
			else:
				skipgram_pairs.append([token_ids[i], clamped_context[j]])

	if len(skipgram_pairs) == 0:
		return torch.empty((0,2), dtype=torch.int64)
	
	return torch.tensor(skipgram_pairs, dtype=torch.int64)