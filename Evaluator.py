# any evaluation function like confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score


class Evaluator:
    @staticmethod
    @staticmethod
    def compute_confusion_matrix(y_actual, y_pred):
        TP = FP = TN = FN = 0
        for actual, pred in zip(y_actual, y_pred):
            if actual == 1 and pred == 1:
                TP += 1
            elif actual == -1 and pred == 1:
                FP += 1
            elif actual == -1 and pred == -1:
                TN += 1
            elif actual == 1 and pred == -1:
                FN += 1
        return np.array([[TN, FN], [FP, TP]])

    @staticmethod
    def overall_accuracy(y_actual, y_pred):
        correct = sum(1 for actual, pred in zip(y_actual, y_pred) if actual == pred)
        total = len(y_actual)
        return correct / total

