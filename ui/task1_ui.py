from ui.ui import *
import tkinter as tk
from tkinter import messagebox, ttk
from data_processor import *
from tkinter import filedialog 
import os 
from models.adaline import *
from models.perceptron import *
class Task1UI(UI):
    def __init__(self,taskTitle):
        super().__init__(taskTitle)
        # Data NEEDED
        self.dataProcessor = None
        self.selected_features = [] # store two selected features 
        self.X_train,self.y_train,self.X_test,self.y_test=None,None,None,None

        dataset_button = ttk.Button(self.root, text="Choose Dataset", command=self.on_click_choose_data)
        dataset_button.grid(row=0, column=0,padx=5 ,sticky="W")

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
        alpha_entry = ttk.Entry(self.input_frame,textvariable=self.learning_rate)
        alpha_entry.grid(row=6, column=0, sticky="W")

        m_label = ttk.Label(self.input_frame, text="Enter Number of Epochs (m):")
        m_label.grid(row=7, column=0, sticky="W")
        self.epochs = tk.IntVar()
        m_entry = ttk.Entry(self.input_frame,textvariable=self.epochs)
        m_entry.grid(row=8, column=0, sticky="W")

        mse_label = ttk.Label(self.input_frame, text="Enter MSE Threshold:")
        mse_label.grid(row=9, column=0, sticky="W")
        self.mse_threshold = tk.DoubleVar()
        mse_entry = ttk.Entry(self.input_frame,textvariable=self.mse_threshold)
        mse_entry.grid(row=10, column=0, sticky="W")

        self.bias_var = tk.BooleanVar()
        bias_checkbox = ttk.Checkbutton(self.input_frame, text="Add Bias", variable=self.bias_var)
        bias_checkbox.grid(row=11, column=0, sticky="W")

        algorithm_label = ttk.Label(self.input_frame, text="Choose Algorithm:")
        algorithm_label.grid(row=12, column=0, sticky="W")

        self.algorithm_var = tk.StringVar(value="Perceptron")
        perceptron_rb = ttk.Radiobutton(self.input_frame, text="Perceptron", variable=self.algorithm_var, value="Perceptron")
        adaline_rb = ttk.Radiobutton(self.input_frame, text="Adaline", variable=self.algorithm_var, value="Adaline")
        perceptron_rb.grid(row=13, column=0, sticky="W")
        adaline_rb.grid(row=13, column=1, sticky="W")

        visualize_button = ttk.Button(self.root, text="Train", command=self.on_click_train)
        visualize_button.grid(row=14, column=0, padx=10, pady=10,sticky="W")

        view_boundary_button = ttk.Button(self.root, text="Predict", command=self.on_click_predict)
        view_boundary_button.grid(row=14, column=1,)

        visualize_button = ttk.Button(self.root, text="Visualize", command=self.on_click_visualize)
        visualize_button.grid(row=15, column=0, padx=10, pady=10,sticky="W")

        view_boundary_button = ttk.Button(self.root, text="View Decision Boundary", command=self.on_click_view_boundary)
        view_boundary_button.grid(row=15, column=1,)

        self.root.mainloop()
    
    def on_click_choose_data(self):
        file=filedialog.askopenfilename(initialdir = os.path.expanduser( os.getcwd()),title = "Select Dataset",filetypes = (("Text files","*.csv"), ("all files","*.*")))
        self.dataProcessor = DataProcessor(file)
        self.dataProcessor.load_data()
        features =  self.dataProcessor.data.columns.drop(["bird category"])
        self.show_features(features)
  
    def on_select_feature(self,feature):
        if feature in self.selected_features:
            self.selected_features.remove(feature)  # Unselect feature if already selected
        elif len(self.selected_features) < 2:
            self.selected_features.append(feature)  # Select feature if less than 2 are selected
        else:
            messagebox.showwarning("Limit Reached", "You can only select two features.")
        print(self.selected_features)

    def show_features(self,features):
        idx=0
        for feature in features:
            chk = tk.Checkbutton(self.input_frame, text=feature,command=lambda f=feature: self.on_select_feature(f))
            chk.grid(row=2, column=idx, sticky="W")
            idx+=1
        print(self.selected_features)

    def on_click_processing(self):
        if self.dataProcessor.data == None:
            messagebox.showwarning("NULL DATA", "Choose your dataset first")
            return
        self.dataProcessor.process_data()

    def on_click_visualize(self):
        print()
        # call funtion visualizer
        # Visualizer.plot_.....

    def on_click_view_boundary(self):
        print()
        #Visualizer.plot_.....
    
    def on_click_train(self):
        self.X_train,self.y_train,self.X_test,self.y_test = self.dataProcessor.split_data()
        if self.algorithm_var.get() == "Perceptron":
            Perceptron().train(X_train,y_train)
        else:
            Adaline().train(X_train,y_train)
        print()
    
    def on_click_predict(self):
        if self.algorithm_var.get() == "Perceptron":
            Perceptron().predict(X_test)
        else:
            Adaline().predict(X_test)




