# POS Information in Word Embedding Partitions
### Aidan San

## Installation
1. pip3 install nltk
2. Download PTB corpus in NLTK
3. Put PTB corpus in nltk_corpus folder
4. pip3 install numpy
5. pip3 install gensim
6. pip3 install torch
7. Download Google News word embeddings and put in data/word_vectors folder

### For Figures:
1. pip3 install seaborn
2. pip3 install jupyter

## Running:
### For figures (produces the charts in the paper):
jupyter notebook
then open figures/figures.ipynb

### For the dataset (produces the train dev and test files):
python3 create_pos_and_word_vectors.py

### For model training (trains and evaluates the neural net):
python3 main.py