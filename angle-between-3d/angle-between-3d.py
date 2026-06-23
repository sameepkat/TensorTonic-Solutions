import numpy as np

def mag(lst: np.array):
	return np.sqrt(np.sum([x** 2 for x in lst]))

def angle_between_3d(v, w):
	v = np.asarray(v)
	w = np.asarray(w)

	v_ = mag(v)
	w_ = mag(w)

	if v_ == 0 or w_ == 0:
		return np.nan

	cos_theta = np.clip(np.sum((v * w)) / (v_ * w_), -1, 1)

	theta = np.arccos(cos_theta)

	return theta