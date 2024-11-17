import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visualizer import *
from ui.ui import *
import tkinter as tk
from tkinter import messagebox, ttk
from data_processor import *
from Evaluator import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from models.backpropagation import *

class Task2UI(UI):
    def __init__(self, taskTitle):
        super().__init__(taskTitle)
        # Data NEEDED
        self.dataProcessor = None
        self.backpropModel = None

        dataset_button = ttk.Button(self.root, text="Choose Dataset", command=self.on_click_choose_data)
        dataset_button.grid(row=0, column=0, columnspan=2, sticky="EW", padx=5, pady=5)

        processing_button = ttk.Button(self.root, text="Processing", command=self.on_click_processing)
        processing_button.grid(row=0, column=2, columnspan=2, sticky="EW", padx=5, pady=5)

        self.input_frame = ttk.LabelFrame(self.root, text="Hyper-Parameters")
        self.input_frame.grid(row=1, column=0, columnspan=4, sticky="EW", padx=10, pady=10)

        feature_label = ttk.Label(self.input_frame, text="Features:")
        feature_label.grid(row=0, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        class_label = ttk.Label(self.input_frame, text="Classes: 3")
        class_label.grid(row=1, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        alpha_label = ttk.Label(self.input_frame, text="Enter Learning Rate (eta):")
        alpha_label.grid(row=2, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        self.learning_rate = tk.DoubleVar()
        self.learning_rate.set(0.01)
        alpha_entry = ttk.Entry(self.input_frame, textvariable=self.learning_rate)
        alpha_entry.grid(row=2, column=2, columnspan=2, sticky="EW", padx=5, pady=2)

        m_label = ttk.Label(self.input_frame, text="Enter Number of Epochs (m):")
        m_label.grid(row=3, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        self.epochs = tk.IntVar()
        self.epochs.set(1000)
        m_entry = ttk.Entry(self.input_frame, textvariable=self.epochs)
        m_entry.grid(row=3, column=2, columnspan=2, sticky="EW", padx=5, pady=2)

        layers_label = ttk.Label(self.input_frame, text="Layers")
        layers_label.grid(row=4, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        self.layers= tk.IntVar()
        self.layers.set(2)
        layers_entry = ttk.Entry(self.input_frame, textvariable=self.layers)
        layers_entry.grid(row=4, column=2, columnspan=2, sticky="EW", padx=5, pady=2)

        self.bias_var = tk.BooleanVar()
        self.bias_var.set(True)
        bias_checkbox = ttk.Checkbutton(self.input_frame, text="Add Bias", variable=self.bias_var)
        bias_checkbox.grid(row=5, column=0, columnspan=4, sticky="W", padx=5, pady=2)

        algorithm_label = ttk.Label(self.input_frame, text="Choose Activation:")
        algorithm_label.grid(row=6, column=0, columnspan=2, sticky="W", padx=5, pady=2)

        self.algorithm_var = tk.StringVar(value="Sigmoid")
        sigmoid_rb = ttk.Radiobutton(self.input_frame, text="Sigmoid", variable=self.algorithm_var, value="Sigmoid")
        tanh_rb = ttk.Radiobutton(self.input_frame, text="Tanh", variable=self.algorithm_var, value="Tanh")
        sigmoid_rb.grid(row=7, column=0, sticky="W", padx=5, pady=2)
        tanh_rb.grid(row=7, column=1, sticky="W", padx=5, pady=2)

        neurons_label = ttk.Label(self.input_frame, text="neurons/hidden layer (if layers >1 : type sperated by space)")
        neurons_label.grid(row=8, column=0, columnspan=2, sticky="W", padx=5, pady=2)
 
        self.neurons= tk.StringVar()
        self.neurons.set("3 4")
        neutons_entry = ttk.Entry(self.input_frame, textvariable=self.neurons)
        neutons_entry.grid(row=8, column=2, columnspan=2, sticky="EW", padx=5, pady=2)

        train_button = ttk.Button(self.root, text="Train", command=self.on_click_train)
        train_button.grid(row=2, column=0, sticky="EW", padx=5, pady=5)

        # predict_button = ttk.Button(self.root, text="Predict", command=self.on_click_predict)
        # predict_button.grid(row=2, column=1, sticky="EW", padx=5, pady=5)

        # view_boundary_button = ttk.Button(self.root, text="View Decision Boundary", command=self.on_click_view_boundary)
        # view_boundary_button.grid(row=2, column=2, sticky="EW", padx=5, pady=5)

        # visualize_btn = ttk.Button(self.root, text="Scatter Train Data", command=self.on_click_visualize)
        # visualize_btn.grid(row=2, column=3, sticky="EW", padx=5, pady=5)
        self.root.mainloop()


    def show_features(self, features):
        idx = 2
        for feature in features:
            chk = tk.Label(self.input_frame, text=feature)
            chk.grid(row=0, column=idx)
            idx += 1

    def on_click_choose_data(self):
        file="dataset/birds.csv"
        self.dataProcessor = DataProcessor(file)
        features = self.dataProcessor.data.columns.drop(["bird category"])
        self.show_features(features)

    def on_click_processing(self):
        if self.dataProcessor == None:
            messagebox.showwarning("NULL DATA", "Choose your dataset first")
            return
        self.dataProcessor.process_data(None,True)

    def on_click_train(self):
        self.backpropModel = BackPropagation(bias=True,epochs=1000,isSigmoid=True,layers=2,learning_rate=0.01,neurons=[4,3])
        # self.backpropModel.train()      
    