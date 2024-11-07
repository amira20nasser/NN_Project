"""
Draw a line that can discriminate between the two learned classes. 
You should also scatter the points of both classes to visualize the behavior of the line (IMPORTANT as We will use it later in documentation).
"""
# ANY plots 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class Visualizer:
    @staticmethod
    def plot_scatter(X , y):
        class_0 = X.loc[y == -1]
        class_1 = X.loc[y == 1]

        plt.Figure(figsize=(5, 4))
        plt.scatter(class_0.iloc[:, 0], class_0.iloc[:, 1], color='blue', label='Class 0')
        plt.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], color='red', label='Class 1')
        # sns.scatterplot(x=X, y=y, hue=y, 
        #                 palette='viridis', style=y)
        plt.title(f'{X.columns[0]} vs {X.columns[1]}')
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.legend(title='Class')
        plt.show()