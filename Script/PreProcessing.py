
import spacy
from spellchecker import SpellChecker


class PreProcess:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.spell = SpellChecker()
    
    def lower_case(self, sentence):
        return sentence.lower()
    
    def tokenization(self, sentence):
        return [word.text for word in self.nlp(sentence) if word.text!= " "]
    
    def spell_check(self, token):
        misspelled = self.spell.unknown(token)
        for word in misspelled:
            token[token.index(word)] = self.spell.correction(word)        
        return " ".join(token)
    
    def filtering(self, sentence):        
        word_list = []
        for word in self.nlp(sentence):
            if word.pos_ not in ["PUNCT"]:
                if word.dep_ not in ["prep", "det"]:
                    if word not in self.stop_words:
                        word_list.append(word)
        return " ".join(map(str, word_list))
    
    def question_demoting(self, question, answer):        
        question_tokens = self.tokenization(question) 
        answer_tokens = self.tokenization(answer)
        demoted_answer = [word for word in answer_tokens if word not in question_tokens]
        return " ".join(demoted_answer)
    
    def lemmantization(self, sentence):
        lemma_words = [word.lemma_ for word in self.nlp(sentence)] 
        return " ".join(lemma_words)    

#     def remove_stop_words(self, sentence):
#         sentence = sentence.split()
#         sentence = [word for word in sentence if not word in self.stop_words]
#         return " ".join(sentence)
        