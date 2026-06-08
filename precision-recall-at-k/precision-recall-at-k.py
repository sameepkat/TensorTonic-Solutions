def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    topk = recommended[:k]
    precision = [item for item in topk if item in relevant]
    precision = len(precision)/k
    

    recall = [item for item in topk if item in relevant]
    recall = len(recall) / len(relevant)

    return [precision, recall]