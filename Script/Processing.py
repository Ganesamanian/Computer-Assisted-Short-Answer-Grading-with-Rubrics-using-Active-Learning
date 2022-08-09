
import spacy
import gensim
import nltk
import re
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import modified_precision
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.translate.bleu_score import modified_precision

from PreProcessing import PreProcess

pre_process = PreProcess()



class Processing:
    
    def __init__(self):
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        self.wmodel = gensim.models.FastText.load('combined_models/combined/fasttext/ft').wv
        self.nlp = spacy.load("en_core_web_lg")
        self.gram = ("NP: {<DT>?<JJ>*<NN>}")
#         self.elmo = hub.Module("https://tfhub.dev/google/elmo/3",
#                                trainable=True)
        
    def cosinesimilarity(self, vector1, vector2):
        return cosine_similarity([vector1], [vector2])
    
    def sentence_embedding(self, sentence):
        sentence_embeddings = self.sbert.encode(sentence)
        return sentence_embeddings
    
    def ngrams(self, refans, ans, n):
        refans_tokens = pre_process.tokenization(refans) 
        ans_tokens = pre_process.tokenization(ans)
        return float(modified_precision(refans_tokens, 
                                        ans_tokens, n))    
    
    def intialpreprocessing(self, sentence):
        sentence1 = pre_process.lower_case(sentence)
        sentence1 = pre_process.tokenization(sentence1)
        sentence1 = pre_process.spell_check(sentence1)
        sentence1 = pre_process.filtering(sentence1)
        return sentence1
        
    def word_count(self, refans, ans):           
        vectorizer = CountVectorizer(binary=True)
        matrix = vectorizer.fit_transform([refans, ans])
        return self.cosinesimilarity(matrix.toarray()[0], 
                                     matrix.toarray()[1])

    def word_embedding(self, refans, ans):        
        if len(ans) != 0:
            return self.cosinesimilarity(self.wmodel.get_vector(refans),
                                         self.wmodel.get_vector(ans))
#             return self.cosinesimilarity(self.nlp(refans).vector, 
#                                          self.nlp(ans).vector)
        else:
            return float(-1)
        
    def word_embedding2(self, refans, ans):        
        if len(ans) != 0:
#             return self.cosinesimilarity(self.wmodel.get_vector(refans),
#                                          self.wmodel.get_vector(ans))
            return self.cosinesimilarity(self.nlp(refans).vector, 
                                         self.nlp(ans).vector)
        else:
            return float(-1)
    
    def lengthratio(self, refans, ans):
           
        return len(ans)/len(refans)
    
#     def elmo_vectors(self, sentence):
#         embeddings = self.elmo(sentence, signature="default", as_dict=True)["elmo"]

#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(tf.tables_initializer())
#             # return average of ELMo features
#             return sess.run(tf.reduce_mean(embeddings,1))
        
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
    