#https://radimrehurek.com/gensim/models/keyedvectors.html
import ptb_reader

from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format("word_vectors/GoogleNews-vectors-negative300.bin", limit=1000000, binary=True)  # C binary format

with open('pos_and_word_vectors.txt', 'w') as f: 
    for word, vec in ptb_reader.get_word_to_posvec().items():
        try:
            f.write(word + "\t" + ",".join([str(x) for x in vec]) + "\t" + ",".join([str(x) for x in wv[word]]) + "\n")
        except:
            print(word)