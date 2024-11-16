# any evaluation function like confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score


class Evaluator:
    @staticmethod
    def compute_confusion_matrix(y_actual, y_pred):
        tp = np.sum((y_actual == 1) & (y_pred == 1))
        tn = np.sum((y_actual == -1) & (y_pred == -1))
        fp = np.sum((y_actual == -1) & (y_pred == 1))
        fn = np.sum((y_actual == 1) & (y_pred == -1))
        cm = np.array([[tn, fp], [fn, tp]])
        return cm

    @staticmethod
    def overall_accuracy(y_actual, y_pred):
        return accuracy_score(y_actual, y_pred)



