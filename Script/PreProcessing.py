# Header
import spacy
from spellchecker import SpellChecker

# Class to preprocess the sentence
class PreProcess():
    """
    It contains preprocessing methods

    """
    
    def __init__(self):
        """
        Initiating the model to be used for 
        preprocessing


        Variables:
            self.nlp (object): spacy model trained on large wikipedia corpus
            self.stop_words (object): contains spacy stopwords
            self.spell (object): spellchecker

        """
        self.nlp = spacy.load("en_core_web_lg")
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.spell = SpellChecker()
    
    def lower_case(self, sentence):
        """
        Function to lower the characters

        Args:
            sentence (string): Continous strings to
                               lower the case

        Returns:
            string: continous strings of lower case
        """
        return sentence.lower()
    
    def tokenization(self, sentence):
        """
        Function to tokenize the sentence

        Args:
            sentence (string): Continous strings to
                               tokenize

        Returns:
            list: list of words in the sentence
        """
        return [word.text for word in self.nlp(sentence) if word.text!= " "]
    
    def spell_check(self, token):
        """
        Function to perform spell check

        Args:
            token (list): list of words in the sentence

        Returns:
            string: continous strings of lower case
        """
        misspelled = self.spell.unknown(token)
        for word in misspelled:
            token[token.index(word)] = self.spell.correction(word)        
        return " ".join(token)
    
    def filtering(self, sentence): 
        """
        Function to filter the punctuation,
        stopwords, preposition and delimiters

        Args:
            sentence (string): Continous strings to
                                filter

        Returns:
           string: continous filtered strings
        """

        word_list = []
        for word in self.nlp(sentence):
            if word.pos_ not in ["PUNCT"]:
                if word.dep_ not in ["prep", "det"]:
                    if word not in self.stop_words:
                        word_list.append(word)
        return " ".join(map(str, word_list))
    
    def question_demoting(self, question, answer): 
        """
        Function to demote the question

        Args:
            question (string): Continous strings of questions
            answer (string): Continous strings of students answer

        Returns:
            string: continous demoted strings
        """

        question_tokens = self.tokenization(question) 
        answer_tokens = self.tokenization(answer)
        demoted_answer = [word for word in answer_tokens if word not in question_tokens]
        return " ".join(demoted_answer)
    
    def lemmantization(self, sentence):
        """
        Function to lemmantize the words

        Args:
            sentence (string): Continous string for which
                               the word to lemmantize

        Returns:
            string: continous lemantized strings
        """

        lemma_words = [word.lemma_ for word in self.nlp(sentence)] 
        return " ".join(lemma_words)    

#     def remove_stop_words(self, sentence):
#         sentence = sentence.split()
#         sentence = [word for word in sentence if not word in self.stop_words]
#         return " ".join(sentence)
        