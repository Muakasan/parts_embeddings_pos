import numpy as np
with open('pos_and_word_vectors.txt') as f:
    l = f.readline()
    word, vec1, vec2 = l.split('\t')
    vec1 = np.fromstring(vec1, sep=',')
    vec2 = np.fromstring(vec2, sep=',')
    print(word, vec1, vec2)