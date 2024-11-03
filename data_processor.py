"""
TASK1: Data Preprocessing and *Splitting dataset*
- Split each class into 30 training samples and 20 testing samples, ensuring that they are randomly selected and non-repeated.
- Load and preprocess the dataset 
"""
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
class DataProcessor:
    @staticmethod
    def load_data():
        #self.file_path = 'dataset/birds.csv'
        data = pd.read_csv('dataset/birds.csv')
        y=pd.DataFrame(data.pop('bird category'))
        X=data
        numeric_cols=['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        print("data\n", data.head())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=34)

        print("train\n", X_train.head())

        #Preprocessing training data                              random_state=34)
        gender_mode=X_train['gender'].mode()[0]
        X_train['gender']=X_train['gender'].fillna(gender_mode)
        X_train['gender']=[0 if x.lower() is "female" else 1 for x in X_train['gender']]

        scaler= Normalizer()
        X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
        print("preprocessed train\n", X_train)

        print("test\n", X_test.head())

        X_test['gender']=X_test['gender'].fillna(gender_mode)
        X_test['gender']=[0 if x.lower() is "female" else 1 for x in X_test['gender']]

        X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])
        print("test\n", X_test)

        #Preprocessing test data
        return X_train, X_test, y_train, y_test
        
   