"""
TASK1: Data Preprocessing and *Splitting dataset*
- Split each class into 30 training samples and 20 testing samples, ensuring that they are randomly selected and non-repeated.
- Load and preprocess the dataset 
"""
import pandas as pd 

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path) 
    
    def process_data():
        print()
    def split_data():
        print()
        #return splitted data x_train ,y_train and so on.....
    def predict(self,X):
        print()
   