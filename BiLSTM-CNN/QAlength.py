# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:11:41 2020

@author: tarun
"""


from data import QAData, Vocabulary
from model import QAModel
import random
from scipy.stats import rankdata
import pickle
import numpy as np


vocabulary = Vocabulary("./data/vocab_all.txt")
embedding_file = "./data/word2vec_100_dim.embeddings"
data = pickle.load(open("./data/dev.pkl",'rb'))
print("Total data length", len(data))
qa_data = QAData()
qa_model = QAModel()
train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(vocabulary))
question_lengths = []
good_answer_lengths = []
bad_answer_lengths = []
c = 0
c1 = 0
total = 0
for i,d in enumerate(data):
    #print("Length of",i,"the input is",len(d))
    #print("Contents of", i,"th input", d)
    good_answer_lengths.append(len(d["good"]))
    bad_answer_lengths.append(len(d["bad"]))
    question_lengths.append(len(d["question"]))
    #indices, answers, question = qa_data.process_data(d)
    #break
    qa_data = QAData()
    predict_model.load_weights('model/cnnlstm_predict_weights_epoch_1.h5')


    print (i, len(d['question']))
    if(len(d['question']) > 7):
        # pad the data and get it in desired format
        indices, answers, question = qa_data.process_data(d)
        total +=1
        # get the similarity score
        sims = predict_model.predict([question, answers])
        n_good = len(d['good'])
        max_r = np.argmax(sims)
        max_n = np.argmax(sims[:n_good])
        r = rankdata(sims, method='max')
        c += 1 if max_r == max_n else 0
        c1 += 1 / float(r[max_r] - r[max_n] + 1)
        precision = c / float(total)#len(data))
        mrr = c1 / float(total)#len(data))
        print ("Precision", precision)
        print ("MRR", mrr)
    #break

precision = c / float(total)#len(data))
mrr = c1 / float(total)#len(data))
print ("Precision", precision)
print ("MRR", mrr)

# print("median_question_lengths",np.median(question_lengths))
# print("median_good_answer_lengths",np.median(good_answer_lengths))
# print("median_bad_answer_lengths", np.median(bad_answer_lengths))
# print()
# print("mean_question_length:", np.array(question_lengths).mean())
# print("mean_good_answer_length:", np.array(good_answer_lengths).mean())
# print("mean_bad_answer_length:", np.array(bad_answer_lengths).mean())
# print()
# print("min_question_length:", np.array(question_lengths).min())
# print("min_good_answer_length:", np.array(good_answer_lengths).min())
# print("min_bad_answer_length:", np.array(bad_answer_lengths).min())
# print()
# print("max_question_length:", np.array(question_lengths).max())
# print("max_good_answer_length:", np.array(good_answer_lengths).max())
# print("max_bad_answer_length:", np.array(bad_answer_lengths).max())