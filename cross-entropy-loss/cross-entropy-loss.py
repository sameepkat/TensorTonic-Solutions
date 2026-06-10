def cross_entropy_loss(y_true, y_pred):
		
	sth = []
	
	for i in range(len(y_true)):
		sth.append(y_pred[i][y_true[i]])

	loss = -np.log(sth)

	return np.mean(loss)