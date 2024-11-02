from ui.task1_ui import *
from data_processor import DataProcessor
def main():
    print("WELCOME TO OUR PROJECT")
    X_train, X_test, y_train, y_test = DataProcessor.load_data()
    task1 = Task1UI("Preceptron and Adaline")
main()

