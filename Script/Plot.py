# Header
import matplotlib.pyplot as plt



# Class for metric calculation and publishing
class Plot():
    """
    This is intend to have all possible plots to 
    be used in the analysis of the project
    """
    
    def line_plot(self, title, labels, xlabel, ylabel, x, y):
        """
        Function used to provide line plot
        
        Args:
            title (str): Title of the plot
            labels (str): Legend/label of the line to 
                          be plotted
            xlabel (str): X-axis title/label
            ylabel (str): Y-axis title/label
            x (list): X-axis data
            y (list): Y-axis data
            
        """
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.plot(x, y, label=labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()
        