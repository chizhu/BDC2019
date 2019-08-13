import fasttext 
w2v = fasttext.train_unsupervised(input=out+"corpus.csv")
w2v.save_model(out+'corpus.fasttext.model')
w2v = fasttext.load_model(out+'corpus.fasttext.model')
word2index = {word: index+1 for index, word in enumerate(w2v.words)}
index2word = {index+1: word for index, word in enumerate(w2v.words)}


def get_embedding_matrix():
    m = np.zeros(shape=(len(index2word)+1, 100))
    for i, w in index2word.items():
        m[i, :] = w2v[w]
    return m
