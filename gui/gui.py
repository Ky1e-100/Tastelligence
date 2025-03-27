import sys
import os
from collections import Counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import tkinter as tk
sys.path.append(os.path.join(os.path.dirname(__file__), '../neural_network'))
import nnchemical as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../deepseek'))
import deepseek as ds

class Graph:
    def __init__(self, frame):
        self.frame = frame

        self.fig = Figure(figsize=(3, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)

        self.ax.pie([], radius=1, autopct='%0.2f%%', shadow=True)

        self.chart = FigureCanvasTkAgg(self.fig, self.frame)
        self.chart.get_tk_widget().pack(side=tk.RIGHT, anchor=tk.E)

    def update(self, values, labels):
        self.ax.clear()
        print(labels)

        autopct = lambda pct: '{:1.1f}%'.format(pct) if pct > 5 else ''

        self.ax.pie(values, radius=1, labels=labels, autopct=autopct, shadow=True)

        self.chart.draw_idle()


class tasteUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Tastelligence")
        self.root.geometry("1600x600")

        # Title Frame ---------------------------------------------------------
        self.title_frame = tk.Frame(self.root)
        self.title_frame.pack(side=tk.TOP, padx=30, pady=30, anchor=tk.NW)

        self.graph = Graph(self.title_frame)

        # Title Label
        self.title_label = tk.Label(self.title_frame, text="Tastelligence", font=("Ariel", 24))
        self.title_label.pack(padx=440, side=tk.TOP)

        # Description Label
        self.desc_label = tk.Label(self.title_frame, text="Please input a chemical Smile, and press 'Upload Smile' to determine taste", font=("Ariel", 14), wraplength=300)
        self.desc_label.pack(padx=300, side=tk.TOP)

        #Frame for the ingredient input ----------------------------------------------------
        self.ingredient_frame = tk.Frame(self.root)
        self.ingredient_frame.pack(side=tk.LEFT, padx=30, pady=30, anchor=tk.NW)

        #Submit button
        self.submit_button = tk.Button(
            self.ingredient_frame,
            text="Upload Ingredient",
            command=self.submit_ingredient,
            font=("Ariel", 14)
        )
        self.submit_button.pack(pady=50, side=tk.TOP)

        # Ingredient Label
        self.ingredient_label = tk.Label(
            self.ingredient_frame,
            text="Input Ingredient:",
            font=("Arial", 14)
        )
        self.ingredient_label.pack(pady=5, side=tk.TOP)

        # Ingredient Entry
        self.ingredient_entry = tk.Entry(self.ingredient_frame, width=50)
        self.ingredient_entry.pack(side=tk.TOP)

        # Ingredient Output Frame
        self.ingredient_output_frame = tk.Frame(self.root)
        self.ingredient_output_frame.pack(side=tk.LEFT, padx=30, pady=30, anchor=tk.NW)

        #Output label
        self.output_label = tk.Label(self.ingredient_output_frame, text="Output", font=("Ariel", 14))
        self.output_label.pack(pady=2, side=tk.TOP)

        #Output field
        self.output_field = tk.Text(
            self.ingredient_output_frame,
            height=10,
            width=30,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.output_field.pack(side=tk.BOTTOM)


        #Smile Frame ------------------------------------------------------
        self.smile_frame = tk.Frame(self.root)
        self.smile_frame.pack(side=tk.LEFT, padx=30, pady=30, anchor=tk.NW)

        # Submit button
        self.smile_button = tk.Button(
            self.smile_frame,
            text="Upload Smile",
            command=self.submit_smile,
            font=("Ariel", 14)
        )
        self.smile_button.pack(pady=50, side=tk.TOP)

        # Label
        self.smile_label = tk.Label(
            self.smile_frame,
            text="Input Smile:",
            font=("Arial", 14)
        )
        self.smile_label.pack(side=tk.TOP)

        # Text Entry
        self.smile_entry = tk.Entry(self.smile_frame, width=50)
        self.smile_entry.pack()

        # Smile Output Frame
        self.smile_output_frame = tk.Frame(self.root)
        self.smile_output_frame.pack(side=tk.LEFT, padx=30, pady=30, anchor=tk.NW)

        # Output label
        self.smile_output_label = tk.Label(self.smile_output_frame, text="Output", font=("Ariel", 14))
        self.smile_output_label.pack(pady=2)

        # Output field
        self.smile_output_field = tk.Text(
            self.smile_output_frame,
            height=10,
            width=30,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.smile_output_field.pack(side=tk.BOTTOM)


    def plot(self):
        ''' Plot a blank pie chart as a place holder '''

    def submit_ingredient(self):
        ingredient = self.ingredient_entry.get()

        # Probably add something here to translate the Smile into the ingredient,
        # then we can provide the taste

        taste = ds.requestSMILES(ingredient)

        # Clear the current text
        self.output_field.delete("1.0", tk.END)

        if taste:
            self.output_field.insert(tk.END, f"{taste}\n")

    def submit_smile(self):
        smile = self.smile_entry.get()

        smile_preds = nn.get_pred(smile)

        self.smile_output_field.delete("1.0", tk.END)

        val = max(smile_preds, key=smile_preds.get)

        if smile_preds:
            self.smile_output_field.insert(tk.END, f"The most prominent taste is: {val}")

        # graph the pie chart
        self.graph.update(smile_preds.values(), smile_preds.keys())


def main():
    root = tk.Tk()
    gui = tasteUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()