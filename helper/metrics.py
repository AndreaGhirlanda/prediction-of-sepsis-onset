import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from typing import Tuple


def accuracy(target: np.ndarray, pred: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate accuracy.

    Args:
        target (np.ndarray): The target values.
        pred (np.ndarray): The predicted values.

    Returns:
        Tuple[float, int, int]: Tuple containing accuracy, number of correct predictions, and total number of predictions.
    """
    correct = pred.round() == target
    correct = correct.sum()
    total = pred.shape[0]
    return (correct / total, correct, total)


def f1(target: np.ndarray, pred: np.ndarray, class_unbalance: float) -> Tuple[float, float]:
    """
    Calculate F1 score.

    Args:
        target (np.ndarray): The target values.
        pred (np.ndarray): The predicted values.
        class_unbalance (float): The class unbalance value.

    Returns:
        Tuple[float, float]: Tuple containing F1 score and weighted F1 score.
    """
    f1 = f1_score(target, pred.round())
    weights = [x * (1 / class_unbalance) if x == 1 else x + 1 for x in target]
    weighted_f1 = f1_score(target, pred.round(), sample_weight=weights)
    return f1, weighted_f1


def roc_curves(target: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, RocCurveDisplay]:
    """
    Generate ROC curves.

    Args:
        target (np.ndarray): The target values.
        pred (np.ndarray): The predicted values.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, RocCurveDisplay]: Tuple containing false positive rate, true positive rate, AUC score, thresholds, and ROC curve display.
    """
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = roc_auc_score(target, pred)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    return fpr, tpr, roc_auc, thresholds, display


def pr_curves(target: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, PrecisionRecallDisplay]:
    """
    Generate precision-recall curves.

    Args:
        target (np.ndarray): The target values.
        pred (np.ndarray): The predicted values.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, PrecisionRecallDisplay]: Tuple containing precision, recall, AUC score, thresholds, and precision-recall curve display.
    """
    precision, recall, threshold = precision_recall_curve(target, pred)
    auc_pr = auc(recall, precision)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    return precision, recall, auc_pr, threshold, display


def conf_matrix(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Generate confusion matrix.

    Args:
        target (np.ndarray): The target values.
        pred (np.ndarray): The predicted values.

    Returns:
        np.ndarray: The confusion matrix.
    """
    conf_matrix = confusion_matrix(target, pred.round())
    return conf_matrix