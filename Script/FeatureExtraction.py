
import numpy as np
import pandas as pd
from PreProcessing import PreProcess
from Processing import Processing

# Object creation
process = Processing()
pre_process = PreProcess()

# Class to extract features
class Featurextraction():
    """
    Class to extract features
    """
    
    def __init__(self, data):
        """
        Function that initiates the feature variables

        Args:
            data (dataframe): Dataframe of dataset

        Variables:
            self.sentence_sim_score (array): Contains array of 
                                             sentence similarity
            self.word_count_score (array): Contains array of 
                                           word overlap count
            self.word_embedding_score (array): Contains array of 
                                               word embedding
            self.length_ratio_score (array): Contains array of 
                                             length ratio
            self.chunck_score (array): Contains array of 
                                       chunck score
            self.lsa_score (array): Contains array of 
                                    lsa score

            self.word_count_score_demoted (array): Contains array of 
                                                   word overlap count
                                                   after demoted
            self.word_embedding_score_demoted (array): Contains array of 
                                                       word embedding
                                                       after demoted
            self.length_ratio_score_demoted (array):Contains array of 
                                                    length ratio
                                                    after demoted
            self.chunck_score_demoted (array): Contains array of 
                                               chunck score
                                               after demoted
            self.lsa_score_demoted (array): Contains array of 
                                            lsa score
                                            after demoted

            
        """
        
        self.sentence_sim_score = np.zeros((len(data), 1))
        self.word_count_score = np.zeros((len(data), 1))
        self.word_embedding_score = np.zeros((len(data), 1))
        self.length_ratio_score = np.zeros((len(data), 1))
        self.chunck_score = np.zeros((len(data), 1))
        self.lsa_score = np.zeros((len(data), 1))

        self.word_count_score_demoted = np.zeros((len(data), 1))
        self.word_embedding_score_demoted = np.zeros((len(data), 1))
        self.length_ratio_score_demoted = np.zeros((len(data), 1))
        self.chunck_score_demoted = np.zeros((len(data), 1))
        self.lsa_score_demoted = np.zeros((len(data), 1))

    def savefeatures(self):
        """
        Function to save the features in csv file
        """

        features = np.hstack((self.sentence_sim_score, self.word_count_score,
                     self.word_embedding_score, self.length_ratio_score,
                     self.chunck_score, self.lsa_score, self.lsa_score_demoted,
                     self.word_count_score_demoted, self.word_embedding_score_demoted,
                     self.length_ratio_score_demoted, self.chunck_score_demoted))

        df = pd.DataFrame(features, columns = ['sentence_sim_score', 'word_count_score',
                                               'word_embedding_score', 'length_ratio_score',
                                               'chunck_score', 'lsa_score', 'lsa_score_demoted',
                                               'word_count_score_demoted', 'word_embedding_score_demoted',
                                               'length_ratio_score_demoted', 'chunck_score_demoted'])

        df.to_csv('Selectedfeatures.csv')
    
    def extract(self):
        """
        Function to extrac the features and save in csv file
        """

        for count in range(len(data['answer'])):
        
       
            # Sentence embedding
            refans_embed = process.sentence_embedding(data['refanswer'][count])
            answer_embed = process.sentence_embedding(data['answer'][count])
            self.sentence_sim_score[count,:] = process.cosinesimilarity(refans_embed, answer_embed)
            
            
            # Initial preprocessing for refanswer, question and solution
            refans_word_count = process.intialpreprocessing(data['refanswer'][count])
            ans_word_count = process.intialpreprocessing(data['answer'][count])
            qns_word_count = process.intialpreprocessing(data['question'][count])
            
            
            refans_wembed = process.word_embedding(pre_process.lemmantization(refans_word_count))
            answer_wembed = process.word_embedding(pre_process.lemmantization(ans_word_count))
            try:
                self.word_embedding_score[count,:] = process.cosinesimilarity(refans_wembed, answer_wembed)
            except:
                self.word_embedding_score[count,:] = 0
                
            
            self.word_count_score[count,:] = process.word_count(pre_process.lemmantization(refans_word_count), 
                                                                  pre_process.lemmantization(ans_word_count))
            
            
            self.length_ratio_score[count,:] = process.lengthratio(pre_process.tokenization(data['refanswer'][count]),
                                                              pre_process.tokenization(data['answer'][count]))
            
            try:
                self.chunck_score[count,:] = process.word_count(process.chunking(data['refanswer'][count]), 
                                                           process.chunking(data['answer'][count]))
                
            except:
                # Chunck score gets empty text due to formulas
                self.chunck_score[count,:] = 0
                
            self.lsa_score[count,:] = process.lsa(pre_process.lemmantization(refans_word_count), 
                                           pre_process.lemmantization(ans_word_count))
            
            # Demoted 
            refans_demoted = pre_process.question_demoting(qns_word_count, refans_word_count)
            ans_demoted = pre_process.question_demoting(qns_word_count, ans_word_count)
            
            refans_lemmma = pre_process.lemmantization(refans_demoted) 
            ans_lemma = pre_process.lemmantization(ans_demoted)
            
            
            refans_wembed = process.word_embedding(refans_lemmma)
            answer_wembed = process.word_embedding(ans_lemma)
            try: 
                self.word_embedding_score_demoted[count,:] = process.cosinesimilarity(refans_wembed, answer_wembed)
            except:
                self.word_embedding_score_demoted[count,:] = 0
            
            self.word_count_score_demoted[count,:] = process.word_count(refans_lemmma, ans_lemma)
                
               
            
            self.length_ratio_score_demoted[count,:] = process.lengthratio(pre_process.tokenization(refans_lemmma),
                                                                     pre_process.tokenization(ans_lemma))
            
            
            self.lsa_score_demoted[count,:] = process.lsa(refans_lemmma, ans_lemma)
            
            try:
                self.chunck_score_demoted[count,:] = process.word_count(process.chunking(refans_lemmma), 
                                                                   process.chunking(ans_lemma))
                
            except:
                self.chunck_score_demoted[count,:] = 0

            print("Completed", count+1, "/", len(data['answer']))

        self.savefeatures()
    
        