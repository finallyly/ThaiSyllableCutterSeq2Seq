#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:calc_vocabulary.py
#   Creator: yuliu1finally@gmail.com
#   Time:02/23/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!

from config import  max_seq_len,min_seq_len,filling_symbol;
import  cPickle;
def make_vocabulary(vocab):
    vocab2 = dict();
    i = 0;
    for k in vocab:
        vocab2[k]=i;
        i+=1;
    vocab2[u'$']=i;
    return vocab2;
if __name__ == "__main__":
    align_copus_file_path = "dataset/trainset.corpus.align.txt";
    vocab_file="dataset/vocab.pkl";
    vocab_source=dict();
    vocab_target=dict();
    with open(align_copus_file_path,"r") as f:
        for line in f:
            line = line.decode("UTF-8");
            if line.startswith("Z:"):
                for uchar in line[2:]:
                    if uchar!=u'\n':
                        if uchar not in vocab_source:
                            vocab_source[uchar]=1;
                        else:
                            vocab_source[uchar]+=1;


            if line.startswith("L:"):
                for uchar in line[2:]:
                    if uchar != u'\n':
                        if uchar not in vocab_target:
                            vocab_target[uchar] = 1;
                        else:
                            vocab_target[uchar] += 1;


    vocab_source = make_vocabulary(vocab_source);
    vocab_target = make_vocabulary(vocab_target);
    print("source vocabulary size is {}".format(len(vocab_source)));
    print("target vocabulary size is {}".format(len(vocab_target)));
    cPickle.dump((vocab_source,vocab_target),open(vocab_file,"wb"));




