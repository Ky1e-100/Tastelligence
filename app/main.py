import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../gui'))
import gui as g

import tkinter as tk

def main():
    root = tk.Tk()
    gui = g.tasteUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()