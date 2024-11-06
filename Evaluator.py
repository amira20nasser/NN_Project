# any evaluation function like confusion_matrix 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Evaluator:
    @staticmethod
    def compute_confusion_matrix(y_actual,y_pred):
        return confusion_matrix(y_actual,y_pred)
        

    @staticmethod
    def overall_accuracy(y_actual,y_pred):
        return accuracy_score(y_actual,y_pred)