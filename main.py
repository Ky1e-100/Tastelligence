import gui.py
import nnchemical
import deepseek
import tkinter as tk

def main():
    root = tk.Tk()
    gui = gui.tasteUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()