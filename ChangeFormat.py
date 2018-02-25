#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:ChangeFormat.py
#   Creator: yuliu1finally@gmail.com
#   Time:02/23/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
from config import max_seq_len,min_seq_len,filling_symbol;
if __name__=="__main__":
    #corpus_fpath="dataset/trainset.crfformat.txt";
    #labeled_corpus_fpath="dataset/trainset.corpus.txt";
    corpus_fpath="dataset/th2.txt";
    labeled_corpus_fpath="dataset/th2.corpus.txt";
    linecount = 0;
    with open(labeled_corpus_fpath,"w") as labeled_corpus_file:
        thaichar=[];
        label=[];
        with open(corpus_fpath,"r") as corpus_file:
            for line in corpus_file:
                line=line.strip();
                col = line.split("\t");
                if len(col)==3:
                     thaichar.append(col[0]);
                     label.append(col[2]);
                else:
                     text="".join(thaichar);
                     labels="".join(label);
                     if text!="":
                        text="Z:"+text;
                        labels="L:"+labels;
                        labeled_corpus_file.write("%s\n"%text);
                        labeled_corpus_file.write("%s\n"%labels);
                        thaichar=[];
                        label=[];




