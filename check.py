import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BaseballStatsAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Baseball Stats Analyzer")

        self.label = tk.Label(root, text="Select CSV File:")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select File", command=self.load_data)
        self.select_button.pack(pady=10)

        self.analyze_button = tk.Button(root, text="Analyze", command=self.analyze_data)
        self.analyze_button.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            messagebox.showinfo("File Selected", f"File selected: {file_path}")
            self.df = pd.read_csv(file_path)

    def analyze_data(self):
        if not hasattr(self, 'df'):
            messagebox.showwarning("Warning", "Please select a CSV file first.")
            return

        # Example: Scatter plot of SLG and R
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sns.regplot(x=self.df['SLG'], y=self.df['R'], data=self.df)
        plt.title('Scatter Plot: SLG vs R')

        # Display the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = BaseballStatsAnalyzerApp(root)
    root.mainloop()
