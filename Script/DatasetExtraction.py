
import pandas as pd

# Path of the root folder
root = "/home/ganesh/Documents/Master Thesis/"

# Dataset path
dataset_path = "Dataset/"

# Class to read the dataset
class Readdataset():
    
    def __init__(self):
        # Location of Mohler dataset
        self.mohler_dataset_path = "Dataset/asag_datasets/mohler/cleaned/"
        self.amr_dataset_path = "Dataset/AMR/"
        
    def mohlerdataset(self):
        '''
        Input:- File location
        output:- One dataframe with the extracted data from the csv files
                 and csv file of the same in the dataset location
        
        '''
        
        # List to keep store all questions 
        # and reference answer
        question = []
        refanswer = []
        
        # Reading questions csv
        mohler_data1 = pd.read_csv(root + self.mohler_dataset_path + "questions.csv", delimiter='\t')
        mohler_data1 = mohler_data1.drop(['Unnamed: 0'], axis=1)
        # Reading answer csv
        mohler_data2 = pd.read_csv(root + self.mohler_dataset_path + "answers.csv", delimiter='\t')
        mohler_data2 = mohler_data2.drop(['Unnamed: 0'], axis=1)        
        
        # Looping through answer datafrane 
        # and adding the corresponding questions 
        for key in mohler_data1['id']:
            for counter in range(len(mohler_data2.index[mohler_data2['id'] == key])):
                # Finding the index with id
                occurance_idx = int(mohler_data1.loc[mohler_data1['id'] == key].index[0])
                question.append(mohler_data1['question'][occurance_idx])
                refanswer.append(mohler_data1['solution'][occurance_idx])
        
        # Add columns from the question csv to answer csv
        mohler_data2.insert(1, "question", question)
        mohler_data2.insert(2, "refanswer", refanswer)
        mohler_data2.to_csv(root + dataset_path + "mohler_joined.csv")
        return mohler_data2
    
    def amrdataset(self):
        '''
        Input:- File location
        output:- One dataframe with the extracted data from the csv files
                 and csv file of the same in the dataset location
        
        '''
        
        # List to keep store all questions 
        # there is no reference answers for it
        question = []
        
        # Reading questions csv
        amr_data1 = pd.read_csv(root + self.amr_dataset_path + "questions.csv", delimiter=',')
        # Reading answer csv
        amr_data2 = pd.read_csv(root + self.amr_dataset_path + "answers.csv", delimiter=',')
        amr_data2['answer'] = amr_data2['answer'].str.replace("YOUR ANSWER HERE", " ")        
        
        # Looping through answer datafrane 
        # and adding the corresponding questions 
        for key in amr_data1['id']:
            for counter in range(len(amr_data2.index[amr_data2['id'] == key])):
                # Finding the index with id
                occurance_idx = int(amr_data1.loc[amr_data1['id'] == key].index[0])
                question.append(amr_data1['question'][occurance_idx])
        
        # Add columns from the question csv to answer csv
        amr_data2.insert(1, "question", question)
        amr_data2.to_csv(root + dataset_path + "amr_joined.csv")
        return amr_data2
        