from gensim.models import Word2Vec
import logging
from gensim.models import word2vec
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
sent=word2vec.Text8Corpus("/home/kesci/work/zhifeng/corpus.csv")
word2vecModel = word2vec.Word2Vec(sent, size=300, window=5, min_count=1,iter=5,
                     sg=1,workers=8)
word2vecModel.save(out+"skip_w2v_all_300.model")

# ##### further train
from gensim.models import word2vec
model = word2vec.Word2Vec.load(out+"skip_w2v_all_300.model")
fout = open(out + "new_corpus.csv",'w')
with open(path+"bytedance_contest.final_2.csv",'r') as fin:
    q_last = ''
    for line in tqdm(fin):
        _,q,_,t = line.strip().split(',')
        if q!=q_last:
            q_last = q
            fout.write(q + '\n')
        fout.write(t + '\n')
fout.close()
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
sent=word2vec.Text8Corpus(out + "new_corpus.csv")
model.build_vocab(sent, update=True)
model.train(sent,total_examples=model.corpus_count, epochs=5)
model.save(out+"new_skip_w2v_all_300.model")
