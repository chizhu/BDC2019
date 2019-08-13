fout = open(out + "corpus.csv",'w')
with open(path+"train_final.csv",'r') as fin:
    q_last = ''
    for line in tqdm(fin):
        _,q,_,t,_ = line.strip().split(',')
        if q!=q_last:
            q_last = q
            fout.write(q + '\n')
        fout.write(t + '\n')
with open(path+"test_final_part1.csv",'r') as fin:
    q_last = ''
    for line in tqdm(fin):
        _,q,_,t = line.strip().split(',')
        if q!=q_last:
            q_last = q
            fout.write(q + '\n')
        fout.write(t + '\n')
fout.close()
"""
corpus.txt格式
// 每行是一条语料 以空格分隔
我 鄂温克  三打底裤  是是
说的 
是对的是  
时代大厦 是对的
是赛事方  说的 
 
"""
