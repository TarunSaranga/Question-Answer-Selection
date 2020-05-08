 # -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:45:44 2020

@author: tarun
"""


from data import QAData, Vocabulary
from model import QAModel
import pickle
import numpy as np
import random
from keras.models import Model
import matplotlib.pyplot as plt

vocabulary = Vocabulary("./data/vocab_all.txt")
embedding_file = "./data/word2vec_100_dim.embeddings"
qa_model = QAModel()
train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(vocabulary))

# layer_outputs = [predict_model.layers[0].output, predict_model.layers[1].output, predict_model.layers[2].layers[0].output, predict_model.layers[2].layers[0].output,
#                  predict_model.layers[2].layers[1].output, predict_model.layers[2].layers[2].get_output_at(0),
#                  predict_model.layers[2].layers[2].get_output_at(1), predict_model.layers[2].layers[3].get_output_at(0),
#                  predict_model.layers[2].layers[3].get_output_at(1), predict_model.layers[2].layers[4].get_output_at(0),
#                  predict_model.layers[2].layers[4].get_output_at(1), predict_model.layers[2].layers[5].output,
#                  predict_model.layers[2].layers[6].output, predict_model.layers[2].layers[7].get_output_at(0),
#                  predict_model.layers[2].layers[7].get_output_at(1), predict_model.layers[2].layers[8].get_output_at(0),
#                  predict_model.layers[2].layers[8].get_output_at(1), predict_model.layers[2].layers[9].get_output_at(0),
#                  predict_model.layers[2].layers[9].get_output_at(1), predict_model.layers[2].layers[10].get_output_at(0),
#                  predict_model.layers[2].layers[10].get_output_at(1), predict_model.layers[2].layers[11].output,
#                  predict_model.layers[2].layers[11].output, predict_model.layers[2].layers[13].get_output_at(0),
#                  predict_model.layers[2].layers[13].get_output_at(1), predict_model.layers[2].layers[14].output]

layer_outputs = [predict_model.layers[2].layers[13].get_output_at(0)]#, predict_model.layers[2].layers[13].get_output_at(1), predict_model.layers[2].layers[14].output]

act_model = Model(inputs = predict_model.inputs, outputs=layer_outputs)

data = pickle.load(open("./data/dev.pkl",'rb'))

print("Total data length", len(data))
qa_data = QAData()
random.shuffle(data)
for i,d in enumerate(data):
    print("i",i)
    indices, answers, question = qa_data.process_data(d)
    #print("indices", indices)
   # print("answers", answers.shape)
    #print("question", question.shape)
    print("Input:", d)
    sims = predict_model.predict([question, answers])
    #print("sims", sims)
    n_good = len(d['good'])
    max_r = np.argmax(sims)
    max_n = np.argmax(sims[:n_good])
    if max_r == max_n:
        print("Correct")
    else:
        print("Wrong")
    lo = act_model.predict([question, answers])
    #print("lo", lo)
    
    plt.figure()
    plt.title(str(i)+" Question")
    plt.plot(lo)
    if(i==5):
        break
    # for i,layer in enumerate(lo):
    #     print(i,"th layer shape:", layer.shape)
    #     plt.figure()
    #     plt.title("Layer")
    #     plt.plot(layer)#,cmap="gray")
    #     break
       
        
    #break