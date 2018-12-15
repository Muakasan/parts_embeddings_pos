#Aidan San
#https://stackoverflow.com/questions/5561614/recursively-convert-all-folder-and-file-names-to-lower-case-or-upper-case
#http://www.nltk.org/howto/corpus.html
#http://universaldependencies.org/tagset-conversion/en-penn-uposf.html
#https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time/43067907

from nltk.corpus import ptb
from nltk.tag.mapping import _UNIVERSAL_TAGS

index_to_tag = _UNIVERSAL_TAGS
tag_to_index = {_UNIVERSAL_TAGS[i]: i for i in range(len(_UNIVERSAL_TAGS))}

def get_word_to_posvec():
    word_to_posvec = {}
    for fileid in ptb.fileids('news'):
        for (word, tag) in ptb.tagged_words(fileid, tagset='universal'):
            if word not in word_to_posvec:
                word_to_posvec[word] = [0]*len(_UNIVERSAL_TAGS)    
            word_to_posvec[word][tag_to_index[tag]] += 1
    return word_to_posvec
