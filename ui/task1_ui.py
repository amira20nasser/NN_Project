import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visualizer import *
from ui.ui import *
import tkinter as tk
from tkinter import messagebox, ttk
from data_processor import *
from tkinter import filedialog
import os
from models.adaline import *
from models.perceptron import *
from Evaluator import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class Task1UI(UI):
    def __init__(self, taskTitle):
        super().__init__(taskTitle)
        # Data NEEDED
        self.dataProcessor = None
        self.selected_features = []  # store two selected features
        self.perceptron = None
        self.adaline = None

        dataset_button = ttk.Button(self.root, text="Choose Dataset", command=self.on_click_choose_data)
        dataset_button.grid(row=0, column=0, padx=5, sticky="W")

        dataset_button = ttk.Button(self.root, text="Processing", command=self.on_click_processing)
        dataset_button.grid(row=0, column=1, sticky="W")

        self.input_frame = ttk.LabelFrame(self.root, text="Hyper-Parameters", padding=(10, 10))
        self.input_frame.grid(row=1, column=0, padx=10, pady=10)

        feature_label = ttk.Label(self.input_frame, text="Select Two Features:")
        feature_label.grid(row=1, column=0, sticky="W")

        class_label = ttk.Label(self.input_frame, text="Select Two Classes:")
        class_label.grid(row=3, column=0, sticky="W")

        self.class_var = tk.StringVar(value="A,B")
        class_options = ["A,B", "A,C", "B,C"]
        class_menu = ttk.OptionMenu(self.input_frame, self.class_var, *class_options)
        class_menu.grid(row=4, column=0, sticky="W", columnspan=3)

        alpha_label = ttk.Label(self.input_frame, text="Enter Learning Rate (eta):")
        alpha_label.grid(row=5, column=0, sticky="W")
        self.learning_rate = tk.DoubleVar()

        self.learning_rate.set(0.015)
        alpha_entry = ttk.Entry(self.input_frame,textvariable=self.learning_rate)
        alpha_entry.grid(row=6, column=0, sticky="W")
        m_label = ttk.Label(self.input_frame, text="Enter Number of Epochs (m):")
        m_label.grid(row=7, column=0, sticky="W")
        self.epochs = tk.IntVar()

        self.epochs.set(2)
        m_entry = ttk.Entry(self.input_frame,textvariable=self.epochs)
        m_entry.grid(row=8, column=0, sticky="W")

        mse_label = ttk.Label(self.input_frame, text="Enter MSE Threshold:")
        mse_label.grid(row=9, column=0, sticky="W")
        self.mse_threshold = tk.DoubleVar()

        self.mse_threshold.set(0.5)
        mse_entry = ttk.Entry(self.input_frame,textvariable=self.mse_threshold)
        mse_entry.grid(row=10, column=0, sticky="W")

        self.bias_var = tk.BooleanVar()
        self.bias_var.set(False)
        bias_checkbox = ttk.Checkbutton(self.input_frame, text="Add Bias", variable=self.bias_var)
        bias_checkbox.grid(row=11, column=0, sticky="W")

        algorithm_label = ttk.Label(self.input_frame, text="Choose Algorithm:")
        algorithm_label.grid(row=12, column=0, sticky="W")

        self.algorithm_var = tk.StringVar(value="Perceptron")
        perceptron_rb = ttk.Radiobutton(self.input_frame, text="Perceptron", variable=self.algorithm_var,
                                        value="Perceptron")
        adaline_rb = ttk.Radiobutton(self.input_frame, text="Adaline", variable=self.algorithm_var, value="Adaline")
        perceptron_rb.grid(row=13, column=0, sticky="W")
        adaline_rb.grid(row=13, column=1, sticky="W")

        train_button = ttk.Button(self.root, text="Train", command=lambda: self.on_click_train())
        train_button.grid(row=14, column=0, padx=10, pady=10, sticky="W")

        predict_button = ttk.Button(self.root, text="Predict", command=lambda: self.on_click_predict())
        predict_button.grid(row=14, column=1, )

        view_boundary_button = ttk.Button(self.root, text="View Decision Boundary", command=self.on_click_view_boundary)
        view_boundary_button.grid(row=15, column=0, )
        
        visualize_btn = ttk.Button(self.root, text="Scatter Train Data", command=self.on_click_visualize)
        visualize_btn.grid(row=16, column=0, )

        self.root.mainloop()

    def on_click_choose_data(self):

        #file=filedialog.askopenfilename(initialdir = os.path.expanduser( os.getcwd()),title = "Select Dataset",filetypes = (("Text files","*.csv"), ("all files","*.*")))
        file="dataset/birds.csv"
        self.dataProcessor = DataProcessor(file)

        features = self.dataProcessor.data.columns.drop(["bird category"])
        self.show_features(features)

    def on_select_feature(self, feature):
        if feature in self.selected_features:
            self.selected_features.remove(feature)  # Unselect feature if already selected
        elif len(self.selected_features) < 2:
            self.selected_features.append(feature)  # Select feature if less than 2 are selected
        else:
            messagebox.showwarning("Limit Reached", "You can only select two features.")
        print("Selected Features: ", self.selected_features)

    def show_features(self, features):
        idx = 0
        for feature in features:
            chk = tk.Checkbutton(self.input_frame, text=feature, command=lambda f=feature: self.on_select_feature(f))
            chk.grid(row=2, column=idx, sticky="W")
            idx += 1

    def on_click_processing(self):
        if self.dataProcessor == None:
            messagebox.showwarning("NULL DATA", "Choose your dataset first")
            return
        self.dataProcessor.process_data(self.class_var.get())

    def on_click_visualize(self):
        Visualizer.plot_scatter(self.dataProcessor.X_train[self.selected_features],self.dataProcessor.y_train)

    def on_click_view_boundary(self):
        X_test, y_test = self.dataProcessor.X_test[self.selected_features], self.dataProcessor.y_test
        class_0 = X_test.loc[y_test == -1]
        class_1 = X_test.loc[y_test == 1]
        plt.scatter(class_0.iloc[:, 0], class_0.iloc[:, 1], color='blue', label='Class 0')
        plt.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], color='red', label='Class 1')
        # adjust bias
        if self.algorithm_var.get() == "Perceptron":
            weights = self.perceptron.weights
            bias=self.perceptron.bias
            plt.title(f'eta:{self.perceptron.learning_rate}, epochs{self.perceptron.epochs}')
        else:
            weights = self.adaline.weights
            bias = self.adaline.bias
            plt.title(f'mse_thresh-> {self.adaline.mse_threshold} eta:{self.adaline.learning_rate} {self.adaline.learning_rate}, epochs{self.adaline.epochs}')
        weights=weights.reshape(2,1)
        
        # print(f"x min {X_test[:,0].min()}, x max {X_test[:,0].max()}")
        # print(f"x train {x1_values}")
        if bias is None:
            bias=0
        x1_values = np.linspace(min(X_test.iloc[:, 0]), max(X_test.iloc[:, 0]), 100)
        x2_values = -(weights[0] * x1_values + bias) / weights[1]
    
        # new_window = tk.Toplevel(self.root)
        # fig, ax = plt.subplots()
  
        
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        plt.legend(title='Class')
        plt.plot(x1_values, x2_values.T)
        plt.show()
        # canvas = FigureCanvasTkAgg(fig, master=new_window)
        # canvas.draw()
        # canvas.get_tk_widget().grid(row=4, column=5, padx=10, pady=10, sticky='nsew')


    def on_click_train(self):
        if self.dataProcessor.X_train is None:
            messagebox.showerror("Not Process Data", "Please Process the Data")
            return
        if self.epochs.get() == 0:
            messagebox.showerror("Empty Fields", "fill the epochs")
            return

        X_train, y_train = self.dataProcessor.X_train[self.selected_features], self.dataProcessor.y_train

        if self.algorithm_var.get() == "Perceptron":
            self.perceptron = Perceptron(self.learning_rate.get(), self.epochs.get(), self.bias_var.get())
            self.perceptron.train(X_train, y_train)
        else:
            self.adaline = Adaline(self.learning_rate.get(), self.epochs.get(), self.mse_threshold.get(),
                                   self.bias_var.get())
            self.adaline.train(X_train, y_train)
        # else:
        #     messagebox.showerror("Empty Fields", "Fill MSE")
            



    def on_click_predict(self):
        X_test, y_test = self.dataProcessor.X_test[self.selected_features], self.dataProcessor.y_test
        y_train = self.dataProcessor.y_train
        X_train = self.dataProcessor.X_train[self.selected_features]
        y_pred = None
        y_pred_train = None
        weights = None
        if self.algorithm_var.get() == "Perceptron":
           y_pred = self.perceptron.predict(X_test)
           y_pred_train = self.perceptron.predict(X_train)
           weights = self.perceptron.weights
        else:
           y_pred = self.adaline.predict(X_test)
           y_pred_train = self.adaline.predict(X_train)
           weights = self.adaline.weights

        # print(f"view confusion shape y_act {y_test.shape} y_pred {y_pred.shape}")   
        cm = Evaluator.compute_confusion_matrix(y_actual=y_test,y_pred=y_pred.T)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        plt.show()
        acc = Evaluator.overall_accuracy(y_actual=y_test,y_pred=y_pred.T)
        acc_train = Evaluator.overall_accuracy(y_actual=y_train,y_pred= y_pred_train.T)    
        messagebox.showinfo("Evluation", f"Train Accuracy {acc_train} \nTest Accuracy {acc}\n with weights {weights}")    




