import tkinter as tk

class UI:
    def __init__(self,taskTitle):
        self.root = tk.Tk()

        self.root.title(taskTitle)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

