import numpy as np

def auc(fpr, tpr):
	fpr = np.asarray(fpr)
	tpr = np.asarray(tpr)
	
	AUC = []
	for i in range(len(fpr)-1):
		AUC.append(0.5 * (tpr[i] + tpr[i+1] )* (fpr[i+1] - fpr[i]))

	return np.sum(np.asarray(AUC))