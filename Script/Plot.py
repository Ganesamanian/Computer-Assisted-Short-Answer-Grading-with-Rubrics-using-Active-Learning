
import matplotlib.pyplot as plt



# Class for metric calculation and publishing
class Plot():
    
    def line_plot(self, title, labels, xlabel, ylabel, x, y):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.plot(x, y, label=labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()
        