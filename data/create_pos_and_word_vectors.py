#https://radimrehurek.com/gensim/models/keyedvectors.html
import ptb_reader
from gensim.models import KeyedVectors
import random

word_posvec = sorted(ptb_reader.get_word_to_posvec().items())
random.Random(17).shuffle(word_posvec)

print("Generated Words and POS Vectors")

wv = KeyedVectors.load_word2vec_format("word_vectors/GoogleNews-vectors-negative300.bin", limit=1000000, binary=True)  # C binary format
print("Loaded Embeddings")

data = []
for word, posvec in word_posvec:
    try:
        data.append((word, wv[word], posvec))
    except:
        print(word)

dev_start = len(data) - 7000
test_start = len(data) - 3500

def elem_to_file(f, elem):
    word, embed, posvec = elem
    f.write(word + "\t" + ",".join([str(x) for x in embed]) + "\t" + ",".join([str(x) for x in posvec]) + "\n")

with open('train.txt', 'w') as trainfile:
    for elem in data[:dev_start]:
        elem_to_file(trainfile, elem)

with open('dev.txt', 'w') as devfile:
    for elem in data[dev_start:test_start]:
        elem_to_file(devfile, elem)

with open('test.txt', 'w') as testfile: 
    for elem in data[test_start:]:
        elem_to_file(testfile, elem)

