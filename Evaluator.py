import numpy as np

class Evaluator:
    @staticmethod
    def compute_confusion_matrix(y_actual, y_pred):
        
        tp = np.sum((y_actual == 1) & (y_pred == 1))  # True Positives
        tn = np.sum((y_actual == -1) & (y_pred == -1))  # True Negatives
        fp = np.sum((y_actual == -1) & (y_pred == 1))  # False Positives
        fn = np.sum((y_actual == 1) & (y_pred == -1))  # False Negatives

     
        cm = np.array([[tn, fp], [fn, tp]])
        return cm

    @staticmethod
    def overall_accuracy(y_actual, y_pred):
        tn, fp, fn, tp = Evaluator.compute_confusion_matrix(y_actual, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
