#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import scipy.optimize

import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt


from sklearn import svm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import mord

# Own Imports
from EvaluationMetrics import Metric
from DatasetExtraction import Readdataset
from PreProcessing import PreProcess
from Plot import Plot
from Processing import Processing
from SupervisedLearning import Supervised_learner
from ActiveLearner import Active_learner
from FeatureExtraction import Featurextraction


# Function to Run the model




def runmodel(features, label, model, iterations, al=True):
    
    acc_list = []
    pc_list = []
    rmse_list =[]
    
    for i in range(iterations):

        X_train, X_test, y_train, y_test = train_test_split(features, label)
        
        if al:
            mit = Active_learner(X_train, X_test, y_train, y_test,
                                 model, 95)
        else:
            mit = Supervised_learner(X_train, X_test, y_train, y_test,
                                     model)
            
        mit_predict = mit.learn()  

        mit_metrics = Metric(y_test, mit_predict)
        pc_list.append(mit_metrics.pearsoncorrelation())
        rmse_list.append(mit_metrics.rmse())
#         acc_list.append(mit_metrics.accuracy())
        
        if i%10 ==0:
            print("After", i+1, "iteration")
            print("Average pearson correlation is", np.mean(pc_list))
            print("Average rmse is", np.mean(rmse_list))
#             print("Average accuracy is", np.mean(acc_list))
    
    if iterations > 1: 
        print("\n")
        print("After", iterations, "iteration")
        print("Average pearson correlation is", np.mean(pc_list))
        print("Average rmse is", np.mean(rmse_list))
    #     print("Average accuracy is", np.mean(acc_list))


    return pc_list, rmse_list, acc_list


df_read = pd.read_csv('Selectedfeatures.csv', delimiter=',')
df_read


# # Training

# In[22]:


features = np.hstack((
                        df_read['sentence_sim_score'].to_numpy().reshape(-1, 1),
                        df_read['word_count_score_demoted'].to_numpy().reshape(-1, 1),
                        df_read['word_embedding_score_demoted'].to_numpy().reshape(-1, 1),
                        df_read['length_ratio_score_demoted'].to_numpy().reshape(-1, 1),
                        df_read['chunck_score_demoted'].to_numpy().reshape(-1, 1),
                        df_read['lsa_score_demoted'].to_numpy().reshape(-1, 1),
))

features[np.isnan(features)] = 0
label = data['score_avg'].to_numpy()


# In[12]:


rl_pc1, rl_rmse1, rl_acc1 = runmodel(features, label, svm.SVR(), 100, al=True)


# In[13]:


rl_pc1, rl_rmse1, rl_acc1 = runmodel(features, label, mord.OrdinalRidge(alpha=1), 100, al=True)


# In[23]:


rl_pc1, rl_rmse1, rl_acc1 = runmodel(features, label, RandomForestRegressor(), 100, al=True)


# In[24]:


rl_pc1, rl_rmse1, rl_acc1 = runmodel(features, label, Ridge(), 100, al=True)


