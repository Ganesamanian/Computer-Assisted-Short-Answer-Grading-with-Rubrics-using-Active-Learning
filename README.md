# Computer Assisted Short Answer Grading with Rubrics using Active Learning

This repository my Master Thesis on Automatic Short Anser Grading ends on November 30th 2022, defended on March 15th 2023. Proposal for the same is available 
[here](https://git.inf.h-brs.de/gkolap2s/computer-assisted-short-answer-grading/-/blob/main/Proposal/KolappanG-MTProposal.pdf)



# Abstract

This thesis investigates the benefit of rubrics for grading short answers using an
active learning mechanism. Automating short answer grading using Natural Language
Processing (NLP) is one of the active research areas in the education domain. This could
save time for the evaluator and invest more time in preparing for the lecture. Most of
the research on short answer grading was treated as a similarity task between reference
and student answers. However, grading based on reference answers does not account for
partial grades and does not provide feedback. Also, the grading is automatic that tries to
replace the evaluator. Hence, using rubrics for short answer grading with active learning
eliminates the drawbacks mentioned earlier.

Initially, the proposed approach is evaluated on the Mohler dataset, popularly used
to benchmark the methodology. This phase is used to determine the parameters for
the proposed approach. Therefore, the approach with the selected parameter exceeds
the performance of current State-Of-The-Art (SOTA) methods resulting in the Pearson
correlation value of 0.63 and Root Mean Square Error (RMSE) of 0.85. The proposed
approach has surpassed the SOTA methods by almost 4%.

Finally, the benchmarked approach is used to grade the short answer based on rubrics
instead of reference answers. The proposed approach evaluates short answers from
Autonomous Mobile Robot (AMR) dataset to provide scores and feedback (formative
assessment) based on the rubrics. The average performance of the dataset results in the
Pearson correlation value of 0.61 and RMSE of 0.83. Thus, this research has proven that
rubrics-based grading achieves formative assessment without compromising performance.
In addition, the rubrics have the advantage of generalizability to all answers.

# Software Requirements 

To replicate this research work following software packages are necessary
1. Python3 == 3.6.13
2. Numpy == 1.19.5
3. Pandas == 1.1.5
4. Seaborn == 0.11.2
5. Plotly == 5.10.0
6. Gensim == 3.8.3
7. SpaCy == 3.4.1
8. NLTK == 3.6.7
9. Sklearn/ Scikit-learn == 0.24.2
10. Scipy == 1.5.4
11. ModAL == 0.4.1
12. Pyspellchecker == 0.6.3
13. Matplotlib == 3.3.4
14. Sentence transformers == 2.2.2
15. Jupyter Notebook == 5.4.1
16. Anaconda == 3.0.0 (optional)


It is advisable to use Anaconda, which makes a constrained environment. The script
for this research work is written in Python following the Python3 format. Gensim
framework is utilized for working with word embeddings where the word embeddings
FastText is used from Metzler [60] work. The spellchecker library is used to check the
spelling. Apart from spell check, other preprocessing is done using the SpaCy toolkit.
Sentence transformer framework is used from Huggingface, which has trained on a large
and diverse dataset of over 1 billion training pairs. NLTK toolkit is used for chunking.
Libraries like Numpy and Scipy are used for computation, like concatenating and storing
the features and finding Pearson correlation. Plotly and Seaborn libraries provide better
visualization along with Matplotlib. Scikit-learn provides machine learning models like
SVM, random forest, and calculate RMSE. modAL is used specifically for the active
learning component.



## How to execute this
Clone this into your local machine:
'''
https://github.com/Ganesamanian/Computer-Assisted-Short-Answer-Grading-with-Rubrics-using-Active-Learning.git
'''

The repository includes
1. Dataset directory contains the three datasets.
2. Script directory contains the script files to execute the code and load file to run
the saved model.
3. Proposal directory contains the Proposal of the Thesis.


### Instructions:
###### Step 1(Script):
For each dataset the script files are kept sepearate. So based on the dataset choose the required script, E.g:- HAR_motionsense.py 

###### Step 2(Path):
Kindly change the root path according to the cloned folder location in your system. Folder path and file path remains the same

###### Step 3(Run):
Now its good to run using the command python3 ASAG.py from terminal inside the script folder. 


####### Disclaimer:
The project requires a computational resource equivalent and above to the Nvidia GeForce
GTX 1650 graphics card with 4GB of video Random Access Memory (RAM) for extracting the features.
Check the machine configuration before executing the code.














