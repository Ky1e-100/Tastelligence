import sys
import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class tasteUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Tastelligence")
        self.root.geometry("800x300")

        #Title Label
        self.title_label = tk.Label(self.root, text="Tastelligence", font=("Ariel", 24))
        self.title_label.pack()

        #Description Label
        self.desc_label = tk.Label(self.root, text="Please input a chemical Smile, and press 'Upload Smile' to determine taste", font=("Ariel", 14))
        self.desc_label.pack()

        #Frame for the button
        #self.controls_frame = tk.Frame(self.root)
        #self.controls_frame.pack(pady=10)

        #Submit button
        self.submit_button = tk.Button(
            text="Upload Smile",
            command=self.submit_smile,
            font=("Ariel", 14)
        )
        self.submit_button.pack(padx=10)

        #Frame for text
        #self.display_frame = tk.Frame(self.root)
        #self.display_frame.pack(padx=5, fill=tk.BOTH, expand=True)

        #Text Field
        #self.text_frame = tk.Frame(self.display_frame)
        #self.text_frame.pack(side=tk.BOTTOM, padx=10, fill=tk.BOTH, expand=True)

        self.smile_label = tk.Label(
            text="Input Smile:",
            font=("Arial", 14)
        )
        self.smile_label.pack(padx=5, anchor=tk.NW)

        self.smile_entry = tk.Entry(self.root, width=50)
        self.smile_entry.pack(padx=5, pady=7, anchor=tk.NW)

        self.output_label = tk.Label(self.root, text="Output", font=("Ariel", 14))
        self.output_label.pack(pady=2)

        self.text_field = tk.Text(
            height=7,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.text_field.pack()


    def submit_smile(self):
        smile = self.smile_entry.get()

        # Probably add something here to translate the Smile into the ingredient,
        # then we can provide the taste

        taste = "some taste function"

        if smile:
            self.text_field.insert(tk.END, f"{smile} Tastes --> like {taste}\n")

def main():
    root = tk.Tk()
    gui = tasteUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()