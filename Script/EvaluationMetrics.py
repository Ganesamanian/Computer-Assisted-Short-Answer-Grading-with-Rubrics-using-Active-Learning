# Header
from sklearn import metrics

from scipy import stats



# Class for metric calculation and publishing
class Metric():
    """
    This class contains possible metrics used for 
    Automatic Short Answer Grading.
    
    """
    
    def __init__(self, y_test, prediction):

        """
        Initiating the class with two array
        
        Args:
            y_test (array): numpy array with actual labels
            prediction (array): numpy array with predicted labels
            
        """        
        self.predict = prediction
        self.y_test = y_test
        
    def pearsoncorrelation(self):
        """
        This function returns the correlation
        using pearson from scipy
        between the actual and predicted scores

        Returns:
            list: list of correlation values
            between actual and predicted scores
        """
        return stats.pearsonr(self.y_test,
                              self.predict)[0]
    
    def rmse(self):

        """
        This function returns the rmse
        using sklearn metrics
        between the actual and predicted scores

        Returns:
            list: list of rmse values
            between actual and predicted scores
        """
        return metrics.mean_squared_error(self.y_test, 
                                          self.predict, 
                                          squared=False)
    
    def accuracy(self):
        """
        This function returns the accuracy
        using sklearn metrics
        between the actual and predicted scores

        Returns:
            list: list of accuracy values
            between actual and predicted scores
        """
        return metrics.accuracy_score(self.y_test, 
                                      self.predict)
    
    def f1score(self):
        """
        This function returns the f1score
        using sklearn metrics
        between the actual and predicted scores

        Returns:
            list: list of f1score values
            between actual and predicted scores
        """
        return metrics.f1_score(self.y_test, 
                               self.predict,
                               average='weighted')
         
    def quadratickappa(self):
        """
        This function returns the quadratickappa values
        using sklearn metrics
        between the actual and predicted scores

        Returns:
            list: list of quadratickappa values
            between actual and predicted scores
        """
        return metrics.cohen_kappa_score(self.y_test, 
                                         self.predict, 
                                         weights = 'quadratic')
    
    def classificationreport(self):

        """
        This function returns the classification report 
        using sklearn metrics
        between the actual and predicted scores

        Returns:
            report: report for classification
            between actual and predicted scores
        """
        return metrics.classification_report(self.y_test, 
                                             self.predict)
        
    def confusionmatrix(self):

        """
        This function returns the Matrix- 2D array 
        using confusion matrix from sklearn metrics
        between the actual and predicted scores

        Returns:
            Matrix-2D array: Display consfusion matrix 
            using seaborn
        """
        
        cf_matrix = metrics.confusion_matrix(self.y_test, 
                                             self.predict)
        plt.figure(figsize=(15,10))
        cm = sns.heatmap(cf_matrix, 
                         annot=True, 
                         cmap='Reds', 
                         fmt='g')
        plt.title("Confusion matrix", fontsize=25)
#         plt.xlabel("Predicted activities", fontsize=20)
#         plt.ylabel("Actual activities", fontsize=20)
        plt.xticks(fontsize= 15, rotation=90)
        plt.yticks(fontsize= 15, rotation=0)
    #     plt.savefig('Motionsense_confusion_matrix_'+title+"_"+str(time_step)+"_"+str(window_size)+'.png',
    #                 bbox_inches = 'tight')
        plt.show()
        