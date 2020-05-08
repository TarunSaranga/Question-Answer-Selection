from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, merge, Lambda
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Convolution1D
from keras.models import Model
import numpy as np

class QAModel():
    def get_cosine_similarity(self):
        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())


    def get_bilstm_model(self, embedding_file, vocab_size):
        """
        Return the bilstm training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05
        enc_timesteps = 150
        dec_timesteps = 150
        hidden_dim = 128

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        weights = np.load(embedding_file)
        qa_embedding = Embedding(input_dim=vocab_size,output_dim=weights.shape[1],mask_zero=True,weights=[weights])
        bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=False))

        # embed the question and pass it through bilstm
        question_embedding =  qa_embedding(question)
        question_enc_1 = bi_lstm(question_embedding)

        # embed the answer and pass it through bilstm
        answer_embedding =  qa_embedding(answer)
        answer_enc_1 = bi_lstm(answer_embedding)

        # get the cosine similarity
        similarity = self.get_cosine_similarity()
        question_answer_merged = merge(inputs=[question_enc_1, answer_enc_1], mode=similarity, output_shape=lambda _: (None, 1))
        lstm_model = Model(name="bi_lstm", inputs=[question, answer], outputs=question_answer_merged)
        good_similarity = lstm_model([question, answer_good])
        bad_similarity = lstm_model([question, answer_bad])

        # compute the loss
        loss = merge(
            [good_similarity, bad_similarity],
            mode=lambda x: K.relu(margin - x[0] + x[1]),
            output_shape=lambda x: x[0]
        )

        # return training and prediction model
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model

