# -*- encoding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape
from keras.initializers import glorot_normal
from keras import regularizers
import keras.layers


class SimpleRNN_combined_time():

    def __init__(self, agent_vocab_size, visitor_vocab_size, maxlen, time_maxlen, dropout, agent_embedding,
                 visitor_embedding, agent_embedding_dim, visitor_embedding_dim):

        self.agent_vocab_size = agent_vocab_size
        self.visitor_vocab_size = visitor_vocab_size
        self.maxlen = maxlen
        self.time_maxlen = time_maxlen
        self.dropout = dropout
        self.agent_embedding_matrix = agent_embedding
        self.visitor_embedding_matrix = visitor_embedding
        self.agent_embedding_dim = agent_embedding_dim
        self.visitor_embedding_dim = visitor_embedding_dim

    def __call__(self):
        agent_text = Input(shape=(self.maxlen,), dtype='int32', name='agent_text')
        visitor_text = Input(shape=(self.maxlen,), dtype='int32', name='visitor_text')
        agent_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='agent_time')
        visitor_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='visitor_time')

        agent_x = Embedding(output_dim=self.agent_embedding_dim, input_dim=self.agent_vocab_size, input_length=self.maxlen,
                            weights=[self.agent_embedding_matrix], trainable=False)(agent_text)
        visitor_x = Embedding(output_dim=self.visitor_embedding_dim, input_dim=self.visitor_vocab_size,
                              input_length=self.maxlen, weights=[self.visitor_embedding_matrix],
                              trainable=False)(visitor_text)
        agent_time_x = Reshape((1, self.time_maxlen))(agent_time)
        visitor_time_x = Reshape((1, self.time_maxlen))(visitor_time)

        agent_output = keras.layers.SimpleRNN(units=self.agent_embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(agent_x)
        visitor_output = keras.layers.SimpleRNN(units=self.visitor_embedding_dim, dropout=self.dropout,
                                          recurrent_dropout=self.dropout)(visitor_x)
        agent_time_output = keras.layers.SimpleRNN(units=1, dropout=self.dropout,
                                                   recurrent_dropout=self.dropout)(agent_time_x)
        visitor_time_output = keras.layers.SimpleRNN(units=1, dropout=self.dropout,
                                               recurrent_dropout=self.dropout)(visitor_time_x)

        merge_dim = self.agent_embedding_dim + self.visitor_embedding_dim + 2
        x = keras.layers.concatenate([agent_output, visitor_output, agent_time_output, visitor_time_output])
        x = Dense(merge_dim, activation='tanh', kernel_initializer=glorot_normal(seed=None))(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=[agent_text, visitor_text, agent_time, visitor_time],
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class GRU_combined():

    def __init__(self, agent_vocab_size, visitor_vocab_size, maxlen, dropout, agent_embedding,
                 visitor_embedding, agent_embedding_dim, visitor_embedding_dim):

        self.agent_vocab_size = agent_vocab_size
        self.visitor_vocab_size = visitor_vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.agent_embedding_matrix = agent_embedding
        self.visitor_embedding_matrix = visitor_embedding
        self.agent_embedding_dim = agent_embedding_dim
        self.visitor_embedding_dim = visitor_embedding_dim

    def __call__(self):
        agent_text = Input(shape=(self.maxlen,), dtype='int32', name='agent_text')
        visitor_text = Input(shape=(self.maxlen,), dtype='int32', name='visitor_text')

        agent_x = Embedding(output_dim=self.agent_embedding_dim, input_dim=self.agent_vocab_size, input_length=self.maxlen,
                            weights=[self.agent_embedding_matrix], trainable=False)(agent_text)
        visitor_x = Embedding(output_dim=self.visitor_embedding_dim, input_dim=self.visitor_vocab_size,
                              input_length=self.maxlen, weights=[self.visitor_embedding_matrix],
                              trainable=False)(visitor_text)

        agent_output = keras.layers.GRU(units=self.agent_embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(agent_x)
        visitor_output = keras.layers.GRU(units=self.visitor_embedding_dim, dropout=self.dropout,
                                          recurrent_dropout=self.dropout)(visitor_x)

        merge_dim = self.agent_embedding_dim + self.visitor_embedding_dim
        x = keras.layers.concatenate([agent_output, visitor_output])
        x = Dense(merge_dim, activation='tanh', kernel_initializer=glorot_normal(seed=None))(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=[agent_text, visitor_text],
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class GRU_combined_time():

    def __init__(self, agent_vocab_size, visitor_vocab_size, maxlen, time_maxlen, dropout, agent_embedding,
                 visitor_embedding, agent_embedding_dim, visitor_embedding_dim):

        self.agent_vocab_size = agent_vocab_size
        self.visitor_vocab_size = visitor_vocab_size
        self.maxlen = maxlen
        self.time_maxlen = time_maxlen
        self.dropout = dropout
        self.agent_embedding_matrix = agent_embedding
        self.visitor_embedding_matrix = visitor_embedding
        self.agent_embedding_dim = agent_embedding_dim
        self.visitor_embedding_dim = visitor_embedding_dim

    def __call__(self):
        agent_text = Input(shape=(self.maxlen,), dtype='int32', name='agent_text')
        visitor_text = Input(shape=(self.maxlen,), dtype='int32', name='visitor_text')
        agent_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='agent_time')
        visitor_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='visitor_time')

        agent_x = Embedding(output_dim=self.agent_embedding_dim, input_dim=self.agent_vocab_size, input_length=self.maxlen,
                            weights=[self.agent_embedding_matrix], trainable=False)(agent_text)
        visitor_x = Embedding(output_dim=self.visitor_embedding_dim, input_dim=self.visitor_vocab_size,
                              input_length=self.maxlen, weights=[self.visitor_embedding_matrix],
                              trainable=False)(visitor_text)
        agent_time_x = Reshape((1, self.time_maxlen))(agent_time)
        visitor_time_x = Reshape((1, self.time_maxlen))(visitor_time)

        agent_output = keras.layers.GRU(units=self.agent_embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(agent_x)
        visitor_output = keras.layers.GRU(units=self.visitor_embedding_dim, dropout=self.dropout,
                                          recurrent_dropout=self.dropout)(visitor_x)
        agent_time_output = keras.layers.GRU(units=1, dropout=self.dropout,
                                                   recurrent_dropout=self.dropout)(agent_time_x)
        visitor_time_output = keras.layers.GRU(units=1, dropout=self.dropout,
                                               recurrent_dropout=self.dropout)(visitor_time_x)

        merge_dim = self.agent_embedding_dim + self.visitor_embedding_dim + 2
        x = keras.layers.concatenate([agent_output, visitor_output, agent_time_output, visitor_time_output])
        x = Dense(merge_dim, activation='tanh', kernel_initializer=glorot_normal(seed=None))(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=[agent_text, visitor_text, agent_time, visitor_time],
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class LSTM_combined_time():

    def __init__(self, agent_vocab_size, visitor_vocab_size, maxlen, time_maxlen, dropout, agent_embedding,
                 visitor_embedding, agent_embedding_dim, visitor_embedding_dim):

        self.agent_vocab_size = agent_vocab_size
        self.visitor_vocab_size = visitor_vocab_size
        self.maxlen = maxlen
        self.time_maxlen = time_maxlen
        self.dropout = dropout
        self.agent_embedding_matrix = agent_embedding
        self.visitor_embedding_matrix = visitor_embedding
        self.agent_embedding_dim = agent_embedding_dim
        self.visitor_embedding_dim = visitor_embedding_dim

    def __call__(self):
        agent_text = Input(shape=(self.maxlen,), dtype='int32', name='agent_text')
        visitor_text = Input(shape=(self.maxlen,), dtype='int32', name='visitor_text')
        agent_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='agent_time')
        visitor_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='visitor_time')

        agent_x = Embedding(output_dim=self.agent_embedding_dim, input_dim=self.agent_vocab_size, input_length=self.maxlen,
                            weights=[self.agent_embedding_matrix], trainable=False)(agent_text)
        visitor_x = Embedding(output_dim=self.visitor_embedding_dim, input_dim=self.visitor_vocab_size,
                              input_length=self.maxlen, weights=[self.visitor_embedding_matrix],
                              trainable=False)(visitor_text)
        agent_time_x = Reshape((1, self.time_maxlen))(agent_time)
        visitor_time_x = Reshape((1, self.time_maxlen))(visitor_time)

        agent_output = keras.layers.LSTM(units=self.agent_embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(agent_x)
        visitor_output = keras.layers.LSTM(units=self.visitor_embedding_dim, dropout=self.dropout,
                                          recurrent_dropout=self.dropout)(visitor_x)
        agent_time_output = keras.layers.LSTM(units=1, dropout=self.dropout,
                                                   recurrent_dropout=self.dropout)(agent_time_x)
        visitor_time_output = keras.layers.LSTM(units=1, dropout=self.dropout,
                                               recurrent_dropout=self.dropout)(visitor_time_x)

        merge_dim = self.agent_embedding_dim + self.visitor_embedding_dim + 2
        x = keras.layers.concatenate([agent_output, visitor_output, agent_time_output, visitor_time_output])
        x = Dense(merge_dim, activation='tanh', kernel_initializer=glorot_normal(seed=None))(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=[agent_text, visitor_text, agent_time, visitor_time],
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class TextCNN_combined_time():

    def __init__(self, agent_vocab_size, visitor_vocab_size, maxlen, time_maxlen, dropout, agent_embedding,
                 visitor_embedding, agent_embedding_dim, visitor_embedding_dim):

        self.agent_vocab_size = agent_vocab_size
        self.visitor_vocab_size = visitor_vocab_size
        self.maxlen = maxlen
        self.time_maxlen = time_maxlen
        self.dropout = dropout
        self.agent_embedding_matrix = agent_embedding
        self.visitor_embedding_matrix = visitor_embedding
        self.agent_embedding_dim = agent_embedding_dim
        self.visitor_embedding_dim = visitor_embedding_dim

    def __call__(self):
        agent_text = Input(shape=(self.maxlen,), dtype='int32', name='agent_text')
        visitor_text = Input(shape=(self.maxlen,), dtype='int32', name='visitor_text')
        agent_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='agent_time')
        visitor_time = Input(shape=(self.time_maxlen,1), dtype='float32', name='visitor_time')

        agent_x = Embedding(output_dim=self.agent_embedding_dim, input_dim=self.agent_vocab_size, input_length=self.maxlen,
                            weights=[self.agent_embedding_matrix], trainable=False)(agent_text)
        visitor_x = Embedding(output_dim=self.visitor_embedding_dim, input_dim=self.visitor_vocab_size,
                              input_length=self.maxlen, weights=[self.visitor_embedding_matrix],
                              trainable=False)(visitor_text)
        agent_time_x = Reshape((1, self.time_maxlen))(agent_time)
        visitor_time_x = Reshape((1, self.time_maxlen))(visitor_time)

        agent_output = keras.layers.GRU(units=self.agent_embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(agent_x)
        visitor_output = keras.layers.GRU(units=self.visitor_embedding_dim, dropout=self.dropout,
                                          recurrent_dropout=self.dropout)(visitor_x)
        agent_time_output = keras.layers.GRU(units=1, dropout=self.dropout,
                                                   recurrent_dropout=self.dropout)(agent_time_x)
        visitor_time_output = keras.layers.GRU(units=1, dropout=self.dropout,
                                               recurrent_dropout=self.dropout)(visitor_time_x)

        merge_dim = self.agent_embedding_dim + self.visitor_embedding_dim + 2
        x = keras.layers.concatenate([agent_output, visitor_output, agent_time_output, visitor_time_output])
        x = Dense(merge_dim, activation='tanh', kernel_initializer=glorot_normal(seed=None))(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=[agent_text, visitor_text, agent_time, visitor_time],
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class SimpleRNN_single():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.SimpleRNN(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class GRU_single():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.GRU(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class GRU_single_time():

    def __init__(self, maxlen, dropout):

        self.maxlen = maxlen
        self.dropout = dropout

    def __call__(self):
        time = Input(shape=(self.maxlen,1), dtype='float32', name='time')
        time_x = Reshape((1, self.maxlen))(time)

        time_output = keras.layers.GRU(units=1, dropout=self.dropout,
                                       recurrent_dropout=self.dropout)(time_x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(time_output)
        model = Model(inputs=time_x,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class LSTM_single():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.LSTM(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class LSTM_single_time():

    def __init__(self, maxlen, dropout):

        self.maxlen = maxlen
        self.dropout = dropout

    def __call__(self):
        time = Input(shape=(self.maxlen,1), dtype='float32', name='time')
        time_x = Reshape((1, self.maxlen))(time)

        time_output = keras.layers.LSTM(units=1, dropout=self.dropout,
                                       recurrent_dropout=self.dropout)(time_x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(time_output)
        model = Model(inputs=time_x,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class TextCNN_single():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.LSTM(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class TextCNN_single_time():

    def __init__(self, maxlen, dropout):

        self.maxlen = maxlen
        self.dropout = dropout

    def __call__(self):
        time = Input(shape=(self.maxlen,1), dtype='float32', name='time')
        time_x = Reshape((1, self.maxlen))(time)

        time_output = keras.layers.LSTM(units=1, dropout=self.dropout,
                                       recurrent_dropout=self.dropout)(time_x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='tanh', name='main_output', activity_regularizer=regularizers.l2(0.001))(time_output)
        model = Model(inputs=time_x,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
