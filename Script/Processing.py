# Header
import spacy
import gensim
import nltk
import re
import numpy as np

from sentence_transformers import SentenceTransformer
# from nltk.translate.bleu_score import modified_precision
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
from scipy.optimize import linear_sum_assignment

from PreProcessing import PreProcess

# Obejct for the preprocessing class
pre_process = PreProcess()


class Processing():
    """
    It contains feature extraction methods
    
    """
    
    def __init__(self):
        
        """
        Initiating the model to be used for embeddings
        and other features


        Variables:
            self.sbert (object): Sentence tranformer model
            self.wmodel (object): Fastext model trained for computer science
            self.nlp (object): spacy model trained on large wikipedia corpus
            self.gram (object): Noun Phrase expression for chunking            

        """
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        self.wmodel = gensim.models.FastText.load('combined_models/combined/fasttext/ft').wv
        self.nlp = spacy.load("en_core_web_lg")
        self.gram = ("NP: {<DT>?<JJ>*<NN>}")
#         self.elmo = hub.Module("https://tfhub.dev/google/elmo/3",
#                                trainable=True)
    
    def intialpreprocessing(self, sentence):
        """
        Function to perform initial preprocessing
        in one go

        Args:
            sentence (string): Continous string which need
                               to be preprocessed 

        Returns:
            string: Continous string after
                    preprocessed
        """
        sentence1 = pre_process.lower_case(sentence)
        sentence1 = pre_process.tokenization(sentence1)
        sentence1 = pre_process.spell_check(sentence1)
        sentence1 = pre_process.filtering(sentence1)
        return sentence1


    def cosinesimilarity(self, vector1, vector2):
        """
        Function to perform cosine similarity
        between two vectors

        Args:
            vector1 (array): Vector or list or array of values/embeddings
            vector2 (array): Vector or list or array of values/embeddings

        Returns:
            list: list of cosine similarity values
                  between two vectors
        """
        return cosine_similarity([vector1], [vector2])
    
    def sentence_embedding(self, sentence):
        """
        Function to calculate sentence embedding
        using sentence transformer

        Args:
            sentence (string): continous string for which embedding
                               to be calculated

        Returns:
            list: list of embeddings
        """
        sentence_embeddings = self.sbert.encode(sentence)
        return sentence_embeddings
    
    # def ngrams(self, refans, ans, n):
    #     refans_tokens = pre_process.tokenization(refans) 
    #     ans_tokens = pre_process.tokenization(ans)
    #     return float(modified_precision(refans_tokens, 
    #                                     ans_tokens, n))    
    
            
    def word_count(self, refans, ans):   
        """
        Function to find word overlap
        between two sentences

        Args:
            refans (string): Continous string of reference answer
            ans (string): Continous string of students answer

        Returns:
            list: list of cosine similarity values
                  between two vectors
        """
                
        vectorizer = CountVectorizer(binary=True)
        matrix = vectorizer.fit_transform([refans, ans])
        return self.cosinesimilarity(matrix.toarray()[0], 
                                     matrix.toarray()[1])

    def word_embedding(self, sentence, sowe=True): 
        """
        Function to calculate word embedding
        using Fastext

        Args:
            sentence (string): Continous string for which embedding
                               to be calculated
            sowe (bool, optional): Flag to do sum operation on 
                                   word embeddings or no. 
                                   Defaults to True.

        Returns:
            float: if sowe is true, gives sum of 
                   word embedding
            list: if sowe is false, gives
                  word embedding
        """
        word_array = []
        for word in pre_process.tokenization(sentence):
            word_array.append(self.wmodel.get_vector(word))

        if sowe:
            return sum(word_array)
        else:
            return word_array
        
    
    def lengthratio(self, refans, ans):
        """
        Function to calculate sentence length ratio

        Args:
            refans (string): Continous string of reference answer
            ans (string): Continous string of students answer

        Returns:
            float: ratio of length
        """
           
        return len(ans)/len(refans)
    
        
    def chunking(self, sentence):
        """
        Function to do chunking
        to get noun phrase

        Args:
            sentence (string): Continous string from
                               which noun phrase should
                               be extracted

        Returns:
            string: Continous string only with
                    noun phrase
        """
    
        word_list = []
        chunking = nltk.RegexpParser(self.gram)        
        sent_token = nltk.word_tokenize(sentence)
        tagging = nltk.pos_tag(sent_token)
        tree = chunking.parse(tagging)
        for tag in tree.pos():
            if tag[1] == "NP":
                word_list.append(tag[0][0])

        return " ".join(word_list)

    def lsa(self, refans, ans):
        """
        Function to calculate
        Linear Sum Assignment

        Args:
            refans (string): Continous string of reference answer
            ans (string): Continous string of students answer

        Returns:
            float: average of the best possible word similarities
        """
    
        we1 = self.word_embedding(refans, sowe=False)
        we2 = self.word_embedding(ans, sowe=False)    
        cost_matrix = np.zeros((len(we1), len(we2)))
        for row, rowvec in enumerate(we1):
            for col, colvec in enumerate(we2):
                cost_matrix[row][col] = self.cosinesimilarity(rowvec, colvec)

        row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)

        sim = [cost_matrix[row][col] for row, col in zip(row_idx, col_idx) if cost_matrix[row][col] >= 0.4]

        return np.mean(sim)
    