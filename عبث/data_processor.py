import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        pddata = pd.read_csv(file_path)
        self.data = pddata
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def select_classes(self, selected_classes):
        if selected_classes == "A,B":
            self.data = self.data[self.data['bird category'].isin(['A', 'B'])]
            self.data['bird category'] = self.data['bird category'].apply(lambda x: 1 if x == 'A' else x)
            self.data['bird category'] = self.data['bird category'].apply(lambda x: -1 if x == 'B' else x)
        if selected_classes == "A,C":
            self.data = self.data[self.data['bird category'].isin(['A', 'C'])]
            self.data['bird category'] = self.data['bird category'].apply(lambda x: 1 if x == 'A' else x)
            self.data['bird category'] = self.data['bird category'].apply(lambda x: -1 if x == 'C' else x)
        if selected_classes == "B,C":
            self.data = self.data[self.data['bird category'].isin(['B', 'C'])]
            self.data['bird category'] = self.data['bird category'].apply(lambda x: 1 if x == 'B' else x)
            self.data['bird category'] = self.data['bird category'].apply(lambda x: -1 if x == 'C' else x)

    def process_data(self, selected_classes, isBackProb):
        y = None
        if isBackProb is None or isBackProb == False:
            self.select_classes(selected_classes)
        else:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_target = encoder.fit_transform(self.data['bird category'].values.reshape(-1, 1))
            encoded_target_df = pd.DataFrame(
                encoded_target,
                columns=[f"Class_{c}" for c in encoder.categories_[0]]
            )
            final_data = pd.concat([self.data.iloc[:, :-1], encoded_target_df], axis=1)
            y = pd.concat([final_data["Class_A"], final_data["Class_B"], final_data["Class_C"]], axis=1)

        X = self.data.drop('bird category', axis=1)
        if y is None:
            y = self.data['bird category'].values

        numeric_cols = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=34, stratify=y)

        gender_mode = self.X_train['gender'].mode()[0]
        self.X_train['gender'] = self.X_train['gender'].fillna(gender_mode)
        self.X_train['gender'] = [0 if x.lower() == "female" else 1 for x in self.X_train['gender']]

        scaler = MinMaxScaler()
        self.X_train[numeric_cols] = scaler.fit_transform(self.X_train[numeric_cols])

        self.X_test['gender'] = self.X_test['gender'].fillna(gender_mode)
        self.X_test['gender'] = [0 if x.lower() == "female" else 1 for x in self.X_test['gender']]
        self.X_test[numeric_cols] = scaler.transform(self.X_test[numeric_cols])

    # Add the get_processed_data() method here to return the processed data
    def get_processed_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

        # print("After Encoding Our Data")
        # print(self.X_train)
        # print(self.y_train)
        # print( (self.y_train[['Class_A','Class_B','Class_C']] == [1, 0, 0]).all(axis=1).sum())
        # print( (self.y_train[['Class_A','Class_B','Class_C']] == [0, 1, 0]).all(axis=1).sum())
        # print( (self.y_train[['Class_A','Class_B','Class_C']] == [0, 0, 1]).all(axis=1).sum())
