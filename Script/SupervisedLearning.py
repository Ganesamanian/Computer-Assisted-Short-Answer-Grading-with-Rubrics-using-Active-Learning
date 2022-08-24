
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.isotonic import IsotonicRegression

import mord


# Class for supervised learning or
# model to run without active learner

class Supervised_learner():
    def __init__(self, X_train, X_test, y_train, y_test, model):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

    def learn(self):
        '''
            Input:- Model, train and test set
            Output:- List of predictions 
        
        '''

        self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(self.X_test)       
        
        return prediction
        