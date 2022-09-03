# Header
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.isotonic import IsotonicRegression

import mord


# Class for supervised learning or
# model to run without active learner

class Supervised_learner():

    """
    This initiate the model with active learning wrapper
    """

    def __init__(self, X_train, X_test, y_train, y_test, model):

        """
        Initiate the class with the model to be used
        along with data to be trained and validated

        Args:
            self.X_train (list): Train data
            self.X_test (list): Test data
            self.y_train (list): Actual label for train data
            self.y_test (list): Actual label for test data
            self.model (class/function): Model under training 
            
            
        """
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

    def learn(self):
        """
        This function trains the model in a supervised manner
        as a old fashioned way

        Returns:
            list: predicted scores
        """
        self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(self.X_test)       
        
        return prediction
        