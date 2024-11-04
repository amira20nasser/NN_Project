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
    def __init__(self, file_path):
        self.file_path = file_path
        pddata = pd.read_csv(file_path)
        self.data = pddata
        self.X_train,self.X_test,self.y_train,self.y_test = None,None,None,None
        
    def process_data(self,selected_classes):
        #self.file_path = 'dataset/birds.csv'
        data = pd.read_csv(self.file_path)
        # y=pd.DataFrame(data.pop('bird category'))
        filtered_data = None
        if selected_classes == "A,B":
            filtered_data = data[(data['bird category'] == 'A') | (data['bird category'] == 'B')]
            filtered_data['bird category'] = [1 if x == "A" else 0 for x in filtered_data['bird category'] ]
        if selected_classes == "A,C":
            filtered_data = data[(data['bird category'] == 'C') | (data['bird category'] == 'A')]
            filtered_data['bird category'] = [1 if x == "A" else 0 for x in filtered_data['bird category'] ]
        if selected_classes == "B,C":
            filtered_data = data[(data['bird category'] == 'B') | (data['bird category'] == 'C')]
            filtered_data['bird category'] = [1 if x == "B" else 0 for x in filtered_data['bird category'] ]

        X = filtered_data.drop('bird category',axis=1)
        y = filtered_data['bird category'].values  
        # X=data
        numeric_cols=['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        # print("data\n", data.head())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4,random_state=34)

        print("train Shape: ", self.X_train.shape)

        #Preprocessing training data                              random_state=34)
        gender_mode=self.X_train['gender'].mode()[0]
        self.X_train['gender']=self.X_train['gender'].fillna(gender_mode)
        self.X_train['gender']=[0 if x.lower() == "female" else 1 for x in self.X_train['gender']]

        scaler= Normalizer()
        self.X_train[numeric_cols]=scaler.fit_transform(self.X_train[numeric_cols])
        # print("preprocessed train\n", self.X_train)

        print("test Shape: ", self.X_test.shape)

        self.X_test['gender']=self.X_test['gender'].fillna(gender_mode)
        self.X_test['gender']=[0 if x.lower() == "female" else 1 for x in self.X_test['gender']]

        self.X_test[numeric_cols]=scaler.transform(self.X_test[numeric_cols])
        # print("test\n", self.X_test)

        #Preprocessing test data
        # return X_train, X_test, y_train, y_test
        
   