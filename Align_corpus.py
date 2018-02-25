#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:Align_corpus.py
#   Creator: yuliu1finally@gmail.com
#   Time:02/23/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
from config import max_seq_len,min_seq_len,filling_symbol;
if __name__=="__main__":
    label_copus_file_path="dataset/th2.corpus.txt";
    align_copus_file_path="dataset/th2.corpus.align.txt";
    with open(align_copus_file_path,"w") as align_copus_file:
        with open(label_copus_file_path,"r") as label_copus_file:
            for line in label_copus_file:
                line = line.strip();
                tag=line[:2];
                line=line[2:];
                uline=line.decode("UTF-8");
                if len(uline)<min_seq_len or len(uline)>max_seq_len:
                    continue;
                line+=filling_symbol*(max_seq_len-len(uline));
                align_copus_file.write("%s%s\n"%(tag,line));
