
import spacy
import gensim
import nltk
import re
from sentence_transformers import SentenceTransformer
# from nltk.translate.bleu_score import modified_precision
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
from scipy.optimize import linear_sum_assignment
import numpy as np

from PreProcessing import PreProcess

pre_process = PreProcess()


class Processing():
    
    def __init__(self):
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        self.wmodel = gensim.models.FastText.load('combined_models/combined/fasttext/ft').wv
        self.nlp = spacy.load("en_core_web_lg")
        self.gram = ("NP: {<DT>?<JJ>*<NN>}")
#         self.elmo = hub.Module("https://tfhub.dev/google/elmo/3",
#                                trainable=True)
    
    def intialpreprocessing(self, sentence):
        sentence1 = pre_process.lower_case(sentence)
        sentence1 = pre_process.tokenization(sentence1)
        sentence1 = pre_process.spell_check(sentence1)
        sentence1 = pre_process.filtering(sentence1)
        return sentence1


    def cosinesimilarity(self, vector1, vector2):
        return cosine_similarity([vector1], [vector2])
    
    def sentence_embedding(self, sentence):
        sentence_embeddings = self.sbert.encode(sentence)
        return sentence_embeddings
    
    # def ngrams(self, refans, ans, n):
    #     refans_tokens = pre_process.tokenization(refans) 
    #     ans_tokens = pre_process.tokenization(ans)
    #     return float(modified_precision(refans_tokens, 
    #                                     ans_tokens, n))    
    
            
    def word_count(self, refans, ans):           
        vectorizer = CountVectorizer(binary=True)
        matrix = vectorizer.fit_transform([refans, ans])
        return self.cosinesimilarity(matrix.toarray()[0], 
                                     matrix.toarray()[1])

    def word_embedding(self, sentence, sowe=True):        
        word_array = []
        for word in pre_process.tokenization(sentence):
            word_array.append(self.wmodel.get_vector(word))

        if sowe:
            return sum(word_array)
        else:
            return word_array
        
    
    def lengthratio(self, refans, ans):
           
        return len(ans)/len(refans)
    
        
    def chunking(self, sentence):
    
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
    
        we1 = self.word_embedding(refans, sowe=False)
        we2 = self.word_embedding(ans, sowe=False)    
        cost_matrix = np.zeros((len(we1), len(we2)))
        for row, rowvec in enumerate(we1):
            for col, colvec in enumerate(we2):
                cost_matrix[row][col] = self.cosinesimilarity(rowvec, colvec)

        row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)

        sim = [cost_matrix[row][col] for row, col in zip(row_idx, col_idx)]

        return np.mean(sim)
    