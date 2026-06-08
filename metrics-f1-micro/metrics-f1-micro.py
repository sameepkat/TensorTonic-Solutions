def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """

    xnor = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            xnor.append(1)
        else:
            xnor.append(0)

    TP = sum(xnor)
    return 2*TP/(2*TP+2*(len(y_true) - TP))