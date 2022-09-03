# Header
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from modAL.models import ActiveLearner
from modAL.density import information_density

import mord


# ## Active learning 
# **With custom query method based on intensity calculated as below for the unlabled dataset $X_u$**
# 
# 
# $$    I(x)=\frac{1}{|X_u|} \sum_{x'\in X} sim(x, x') $$
# 
# $sim(x, x')$ is a cosine similarity function
# 

# In[3]:


# Class for active learning wrapper
class Active_learner():
    """
    This initiate the active learner
    """
    
    # Class variables
    def __init__(self, X_train, X_test, y_train, y_test, model, percentage):

        """
        Initiate the class with the model to be used
        along with data to be trained and validated

        Args:
            self.X_train (list): Train data
            self.X_test (list): Test data
            self.y_train (list): Actual label for train data
            self.y_test (list): Actual label for test data
            self.model (class/function): Model under training 
            self.percentage (int): Percentage of data to query
            
        """
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model         
        self.percent = percentage
#         self.query_method = query_method
    
    # Custom query function
    def intensity_query(self, learner, X):
        """
        Function to query a data
        from the pool of data

        Args:
            learner (class/function): Active learner initiated with
                                      train data and model
            X (list): pool of data to query from

        Returns:
            tuple: query index as int,
                   queried data as list of strings
        """
        
        cosine_density = information_density(X, 'cosine')
        query_idx = np.argmax(cosine_density)        
        return query_idx, X[query_idx]
    
    # Active learning function
    def learn(self):

        """
        This function trains the model in a active learning
        loop with custome query function 

        Returns:
            list: predicted scores
        """
        
        '''
            Input:- Model, train and test set, query method, 
                    percentage of data for training AL
            
        '''
        n_queries = int(len(self.X_train)*((100-self.percent)/100))
        
        initial_idx = np.random.choice(range(len(self.X_train)), 
                                       size=(len(self.X_train)-n_queries), 
                                       replace=False)
        
        # seeding, getting initial data for training
#         unique_label = list(set(self.y_train))
#         initial_idx = [np.where(self.y_train == i)[0][0] for i in unique_label]

        # initialising data for training
        X_initial = self.X_train[initial_idx]
        y_initial = self.y_train[initial_idx]

        # generating the pool
        X_pool = np.delete(sereturns the rmse
        using sklearn metrics
        between the actual and predicted scoreslf.X_train, initial_idx, axis=0)
        y_pool = np.delete(self.y_train, initial_idx, axis=0)
        
#         print(len(X_train), len(X_initial), len(X_pool), len(X_test))

        # initializing the active learner
        learner = ActiveLearner(
            estimator = self.model,
            query_strategy = self.intensity_query,
            X_training = X_initial, y_training = y_initial            
        )
        
        # Calculating initial accuracy with initial data
        accuracy_scores = [learner.score(self.X_test, self.y_test)]
        
        # pool-based sampling
        
#         print("It is calculated minimum", n_queries, "queries are required")
#         print("How you want to proceed?\n 1. Manual\n 2. Automatic")
#         decision_number = np.array([int(input())], dtype=int)
        decision_number = 2
        
        # Query loop
        for counter in range(n_queries):
            
#             display.clear_output(wait=True)
            query_idx, query_instance = learner.query(X_pool)
            # Accuracy plot
#             with plt.style.context('seaborn-white'):
#                 plt.figure(figsize=(10, 5))
#                 plt.title('Accuracy of your model')
#                 plt.plot(range(counter+1), accuracy_scores, label="Accuracy")
#                 plt.scatter(range(counter+1), accuracy_scores)
#                 plt.xlabel('Number of queries')
#                 plt.ylabel('Accuracy')
#                 plt.grid()
#                 plt.legend()
#                 display.display(plt.gcf())
#                 plt.close('all')
            
            if decision_number == 1:
                print("What label is this?")
                y_new = np.array([int(input())], dtype=int)
            else:
                y_new = y_pool[query_idx].reshape(1, )
            
            learner.teach(query_instance.reshape(1, -1), y_new)
            
            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
#             print(len(X_pool), len(y_pool))
            

            accuracy_scores.append(learner.score(self.X_test, self.y_test))
        
#         print("By just labelling ", n_queries, " of total data, accuracy of ", round(accuracy_scores[-1]-accuracy_scores[0], 3), " % is achieved on the unseen data" )
        prediction = learner.predict(self.X_test)
        return prediction




