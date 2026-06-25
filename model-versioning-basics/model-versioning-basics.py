def promote_model(models):

	promotion = models[0]

	for next_model in models:
		if next_model["accuracy"] > promotion["accuracy"]:
			promotion = next_model
			continue
		elif promotion["accuracy"] == next_model["accuracy"] and next_model["latency"] < promotion["latency"]:
			promotion = next_model
			continue
		elif promotion["accuracy"] == next_model["accuracy"] and promotion["latency"] == next_model["latency"] and next_model["timestamp"] > promotion["timestamp"]:
			promotion = next_model
			continue
		
	return promotion["name"]