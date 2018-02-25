#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:FormatDataSet.py
#   Creator: yuliu1finally@gmail.com
#   Time:02/23/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import cPickle;
import numpy as np;
if __name__ =="__main__":
    aligned_corpus_path='dataset/th2.corpus.align.txt';
    dataset_file="dataset/th2dataset.pkl";
    dataset = list();
    src_len = [];
    target_len = [];
    with open(aligned_corpus_path,'r') as f:
        for line in f:
            line = line.decode('utf-8')[:-1]  # remove \n
            if line.startswith('Z:'):
                source = line[2:]
            if line.startswith('L:'):
                target = line[2:]
                src_len.append(len(source));
                target_len.append(len(target));
                dataset.append((source, target));
    cPickle.dump(dataset, open(dataset_file, 'wb'))
    print 'source length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(src_len), max=np.max(src_len),
                                                                 a=np.mean(src_len))
    print 'target length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(target_len), max=np.max(target_len),
                                                                 a=np.mean(target_len))
