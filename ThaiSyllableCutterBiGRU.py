#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:ThaiSyllableCutterBiGRU.py
#   Creator: yuliu1finally@gmail.com
#   Time:02/23/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import cPickle

from config import filling_symbol,min_seq_len,max_seq_len

def join_dicts(dict_list):
    """
    Raise exception if two dicts share some keys
    :param dict_list:
    :return:
    """
    dict_ret = dict();
    for d in dict_list:
        for k,v in d.iteritems():
            if k not in dict_ret:
                dict_ret[k]=v;
            else:
                raise  Exception('Key conflicts in join_dicts');
    return dict_ret;

def weight_variable_normal(shape,stddev=None):
    if stddev is None:
        stddev = 1.0/np.sqrt(shape[0]);
    initial = tf.truncated_normal(shape=shape,mean=0.0,stddev=stddev);
    return tf.Variable(initial);

def weight_variable_uniform(shape,radius=None):
    if radius is None:
        radius = 1.0/np.sqrt(shape[0]);
    initial = tf.random_uniform(shape=shape,minval=-radius,maxval=radius);
    return  tf.Variable(initial);

class GRUCell(object):
    #initialize, tell the data member of the GRUCell class
    def __init__(self,n_input,n_hidden,stddev=None,variable_values=None,name="GRU"):
        if variable_values is None:
            #update gate
            self.W_z = weight_variable_uniform([n_input+n_hidden,n_hidden],stddev);
            self.b_z = tf.Variable(tf.zeros(n_hidden,tf.float32));
            # reset gate
            self.W_r = weight_variable_uniform([n_input + n_hidden, n_hidden], stddev);
            self.b_r = tf.Variable(tf.zeros(n_hidden,tf.float32));
            #Candidate generation
            self.W_c = weight_variable_uniform([n_input+n_hidden,n_hidden],stddev);
            self.b_c = tf.Variable(tf.zeros(n_hidden,tf.float32));
        else:
            self.W_z = tf.Variable(variable_values[":".join([name,"W_z"])]);
            self.b_z = tf.Variable(variable_values[":".join(name,"b_z")]);
            self.W_r = tf.Variable(variable_values[":".join([name,"W_r"])]);
            self.b_r = tf.Variable(variable_values[":".join([name,"b_r"])]);
            self.W_c = tf.Variable(variable_values[":".join([name,"W_c"])]);
            self.b_c = tf.Variable(variable_values[":".join(name,"b_c")]);
        self.n_input = n_input;
        self.n_hidden = n_hidden;
        self.variables={":".join([name,"W_z"]):self.W_z,
                        ":".join([name, "b_z"]):self.b_z,
                        ":".join([name,"W_r"]):self.W_r,
                        ":".join([name, "b_r"]): self.b_r,
                        ":".join([name, "W_c"]): self.W_c,
                        ":".join([name, "b_c"]): self.b_c
                        }


    def __call__(self,h,x):
        hx = array_ops.concat([h,x],1);
        # z: update gate
        z=tf.sigmoid(tf.matmul(hx,self.W_z)+self.b_z);
        #r:reset gate
        r=tf.sigmoid(tf.matmul(hx,self.W_r)+self.b_r);
        #h_c: candidate
        h_candidate=tf.sigmoid(tf.matmul(array_ops.concat([r*h,x],1),self.W_c)+self.b_c);
        new_h = (1-z)*h+z*h_candidate;
        return new_h;

def build_encoder_layers(input,n_step_input,encoder_layers,reverse_input=False):
    """

    :param input: n_samplexn_input_stepxn_input
    :param n_step_input:
    :param encoder_layers:
    :param reverse_input:
    :return:
    """
    n_sample = tf.shape(input)[0];
    n_layer = len(encoder_layers);
    input_list=tf.unstack(input,axis=1);
    if reverse_input:input_list=input_list[::-1];
    states_layers=[];
    for l in range(n_layer):
        states_prev = input_list if l==0 else states_layers[-1];
        encoder = encoder_layers[l];
        h_init = tf.zeros((n_sample,encoder.n_hidden),tf.float32);
        states=[];
        for t in range(n_step_input):
            h_prev = h_init if t==0 else states[-1];
            input_t = states_prev[t];
            h_t=encoder(h_prev,input_t);
            states.append(h_t);
        states_layers.append(states);
    return states_layers;

def vectorise(string, vocab):
    coding = np.zeros((len(string),len(vocab)),np.float32);
    for k,x in zip(range(len(string)),string):
        coding[k,vocab[x]]=1;
    return  coding;

def vectoriseForOtherTest(string, vocab):
    coding = np.zeros((len(string),len(vocab)),np.float32);
    for k,x in zip(range(len(string)),string):
        if x in vocab:
            coding[k,vocab[x]]=1;
        else:
            coding[k,vocab[u'$']]=1;
    return  coding;
def digitalize(string,vocab):
    return  np.array([vocab[x] for x in string]);
def digitalizeForOtherTest(string,vocab):
    v=list();
    for x in string:
        if x in vocab:
            v.append(vocab[x]);
        else:
            v.append(vocab[u'$']);
    return np.array(v);

def prepare_dataForOtherTest(pairs, vocab_source,vocab_target):
    """
    :param pairs:
    :param vocab_source:
    :param vocab_target:
    :return:
    """
    source = [];
    target = [];
    for s, t in pairs:
        source.append(digitalizeForOtherTest(s, vocab_source));
        target.append(vectoriseForOtherTest(t, vocab_target));
    source = np.stack(source, axis=0);
    target = np.stack(target, axis=0);
    return source, target;

def prepare_data(pairs, vocab_source, vocab_target):
    """
    :param pairs: The strings in pairs must be aligned, that is their lengths are padded to the same
    :param vocab_source:
    :param vocab_target:
    :return:
    """
    source = [];
    target = [];
    for s,t in pairs:
        source.append(digitalize(s,vocab_source));
        target.append(vectorise(t,vocab_target));
    source = np.stack(source,axis=0);
    target = np.stack(target,axis=0);
    return source,target;

def main():
    print"xxxxxxxxxxxxxx"
    np.random.seed(1001);
    dataset_file="dataset/dataset_split.pkl";
    vocab_file="dataset/vocab.pkl";
    testdataset_file = "dataset/th2dataset.pkl";
    source_vocab_size=83;
    embed_dim=166;
    n_output=6;
    n_step_input=max_seq_len;
    n_hidden=[256,256];
    n_layer=len(n_hidden);
    weight_stddev=0.1;
    n_epoch = 10;
    batch_size = 100;
    validation_steps = 100;
    save_param_steps = 100;
    learning_rate = 1e-2;
    gamma = 1.0;
    verbose = False;

    #-- build the graph
    x = tf.placeholder(tf.int32,[None,n_step_input],name='x');
    y = tf.placeholder(tf.float32,[None,n_step_input,n_output],name='y');

    #embedding layer
    embedding = weight_variable_uniform([source_vocab_size,embed_dim],radius=weight_stddev);
    embed_x = tf.nn.embedding_lookup(embedding,x);

    encoder_layers = [];
    encoder_r_layers =[];
    variables = dict();
    for l in range(n_layer):
        input_size = embed_dim if l==0 else n_hidden[l-1];
        layer_size = n_hidden[l];
        encoder = GRUCell(input_size,layer_size,weight_stddev,name='encoder:{}'.format(l));
        encoder_r =GRUCell(input_size,layer_size,weight_stddev,name='encoder_r:{}'.format(l));
        variables = join_dicts([variables,encoder.variables,encoder_r.variables]);
        encoder_layers.append(encoder);
        encoder_r_layers.append(encoder_r);
    W_o = weight_variable_uniform([2*n_hidden[-1],n_output],weight_stddev);
    b_o = tf.Variable(np.zeros(n_output,dtype=np.float32));
    variables = join_dicts([variables,{'W_o':W_o,'b_o':b_o,'embedding':embedding}]);

    #encoding
    n_sample = tf.shape(x)[0];
    states_layers = build_encoder_layers(embed_x,n_step_input,encoder_layers,reverse_input=False);
    states_r_layers=build_encoder_layers(embed_x,n_step_input,encoder_r_layers,reverse_input=True);

    #decoding
    outputs = list();
    for  t in range(n_step_input):
        h_t = tf.concat([states_layers[-1][t],states_r_layers[-1][-t-1]],1);
        out_t = tf.nn.softmax(tf.matmul(h_t,W_o)+b_o);
        outputs.append(out_t);
    # outputs: n_sample x n_step x n_output
    outputs = tf.stack(outputs,axis=1)
    # predictions: n_samples x n_step
    predictions = tf.argmax(outputs,axis=2,name='predictions');

    #loss
    loss = -tf.reduce_sum(tf.log(outputs)*y)/(tf.cast(n_sample,tf.float32)*n_step_input);

    #l2-norm of parameters
    regularizer = 0.0;
    for k,v in variables.iteritems():
        regularizer += tf.reduce_mean(tf.square(v));
    regularizer /= len(variables);

    #cost
    cost = loss + gamma*regularizer;
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost);

    init_vars = tf.global_variables_initializer();

    #--run the graph --
    vocab_source, vocab_target = cPickle.load(open(vocab_file,"rb"));
    vocab_source_r = dict();
    for k,v in vocab_source.iteritems():
        vocab_source_r[v]=k;
    vocab_target_r = dict();
    for k,v in vocab_target.iteritems():
        vocab_target_r[v] = k;

    train_set, validation_set, test_set = cPickle.load(open(dataset_file,"rb"));
    testset2=cPickle.load(open(testdataset_file,"rb"));
    print '{tr} training samples, {v} validation samples, {te} test samples, {t}' \
          ' another test samples'.format(tr=len(train_set), v=len(validation_set), te=len(test_set),t=len(testset2));
    n_sample=len(train_set);

    sess = tf.Session();
    with sess.as_default():
        init_vars.run();
        sample_counter = 0;
        for i in range(int(n_epoch*n_sample/batch_size)):
            if i % int(validation_steps) == 0:
                source, target = prepare_data(validation_set,vocab_source,vocab_target);
                c,l,r = sess.run([cost,loss,regularizer],feed_dict={x:source,y:target});
                print '{i} samples fed in: validation:{n} samples, cost {c:.5f},loss{l:.5f},' \
                      'parameter regularizer {r:.5f}'.format(i=sample_counter,n=len(validation_set),c=c,l=l,r=r);

            if i % int(save_param_steps) == 0:
                parameters = dict();
                for k,v in variables.iteritems():
                    parameters[k]=sess.run(v);
                cPickle.dump(parameters,open("models/parameters_{}.pkl".format(i),'wb'));


            selected_idx = np.random.permutation(n_sample)[0:batch_size];
            batch_pairs = [train_set[k] for k in selected_idx];
            source,target = prepare_data(batch_pairs,vocab_source,vocab_target);
            _,c,l,r = sess.run([train_step,cost, loss,regularizer],feed_dict={x:source,y:target});

            if verbose:
                print '{i}-th batch, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(i=i, c=c, l=l,r=r)
            sample_counter += len(batch_pairs);

        parameters=dict();
        for k,v in variables.iteritems():
            parameters[k]=sess.run(v);
        cPickle.dump(parameters,open('models/parameters_final.pkl','wb'));

        #evaluate on test set
        source,target = prepare_data(test_set,vocab_source,vocab_target);
        l=sess.run(loss,feed_dict={x:source,y:target});
        print 'test set:{n} samples, loss{l:.8f}'.format(n=len(test_set),l=l);
    sess.close();

    parameters = cPickle.load(open('models/parameters_final.pkl', 'rb'));
    source,target = prepare_dataForOtherTest(testset2,vocab_source,vocab_target);
    sess = tf.Session();
    feed_dict = dict();
    for k,v in variables.iteritems():
        feed_dict[v]=parameters[k];
    feed_dict[x] = source;
    feed_dict[y] = target;
    l,pred = sess.run([loss,predictions],feed_dict=feed_dict);
    print 'test set: {n} samples, loss {l:.8f}'.format(n=len(testset2), l=l);
    targets = [a[1] for a in testset2];
    texts =[a[0] for a in testset2];
    fout=open('dataset/testMe.txt','w');
    for k in range(0,len(texts)):
        text= texts[k].replace("#",'');
        t=targets[k].replace('#','');
        p=pred[k];
        p=u"".join([vocab_target_r[d] for d in p]);
        p=p.replace('#','');
        fout.write("%s\t%s\t%s\n"%(text.encode('UTF-8'),t.encode("UTF-8"),p.encode("UTF-8")));


if __name__ == '__main__':
    main()






