
from sklearn import metrics
from scipy import stats



# Class for metric calculation and publishing
class Metric():
    
    def __init__(self, y_test, prediction):
        
        self.predict = prediction
        self.y_test = y_test
        
    def pearsoncorrelation(self):
        return stats.pearsonr(self.y_test,
                              self.predict)[0]
    
    def rmse(self):
        return metrics.mean_squared_error(self.y_test, 
                                          self.predict, 
                                          squared=False)
    
    def accuracy(self):
        return metrics.accuracy_score(self.y_test, 
                                      self.predict)
    
    def f1score(self):
        return metrics.f1_score(self.y_test, 
                               self.predict,
                               average='weighted')
         
    def quadratickappa(self):
        return metrics.cohen_kappa_score(self.y_test, 
                                         self.predict, 
                                         weights = 'quadratic')
    
    def classificationreport(self):
        return metrics.classification_report(self.y_test, 
                                             self.predict)
        
    def confusionmatrix(self):
        
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
        