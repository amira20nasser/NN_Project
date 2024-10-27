"""
TASK1: Data Preprocessing and *Splitting dataset*
- Split each class into 30 training samples and 20 testing samples, ensuring that they are randomly selected and non-repeated.
- Load and preprocess the dataset 
"""

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
   