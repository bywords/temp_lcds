# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import writer
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from reader import load_data, load_label, load_data_separate
from preprocessing import filter_labeled_data, transform_sequence, transform_label, transform_time,\
    transform_labeled_data_listform, random_oversampling, get_wordvectors_from_keyedvectors, \
    filter_labeled_data_separate, random_oversampling_separate, transform_labeled_data_listform_separate
from word_embedding import load_embedding
from samsung_rnn import *
from utils import ModelType, EmbeddingType


def read_train_eval(testid, allDataForEmbed, preprocess, maxseq, modelType,
                    dropout, earlyStop, seedNum, batchSize, maxEpoch):
    LOG_BASE_DIR = 'log_separate'
    TRAIN_INSTANCE_DIR = os.path.join(LOG_BASE_DIR, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'
                                      .format(testid, allDataForEmbed, preprocess, maxseq, modelType,
                                              dropout, earlyStop, seedNum, batchSize, maxEpoch))
    if not os.path.isdir(TRAIN_INSTANCE_DIR):
        os.mkdir(TRAIN_INSTANCE_DIR)
    log_csvfile = os.path.join(TRAIN_INSTANCE_DIR, 'log.csv')
    result_file = os.path.join(TRAIN_INSTANCE_DIR, 'results.txt')

    time_maxseq = maxseq / 2
    print('Load data')
    agent_sequence, visitor_sequence, agent_time_sequence, visitor_time_sequence, agent_maxtime, visitor_maxtime = \
        load_data_separate(preprocess=preprocess, maxseq=maxseq)
    label_data = load_label()
    agent_sequence, visitor_sequence, agent_time_sequence, visitor_time_sequence, labels = \
        filter_labeled_data_separate(agent_sequence, visitor_sequence,
                                     agent_time_sequence, visitor_time_sequence, label_data)

    print('Load embedding')
    if preprocess:
        if allDataForEmbed:
            agent_w2v_model = load_embedding(embeddingType=EmbeddingType.PRE_ALL)
            visitor_w2v_model = load_embedding(embeddingType=EmbeddingType.PRE_ALL)
        else:
            agent_w2v_model = load_embedding(embeddingType=EmbeddingType.PRE_AGENT)
            visitor_w2v_model = load_embedding(embeddingType=EmbeddingType.PRE_VISITOR)
    else:
        if allDataForEmbed:
            agent_w2v_model = load_embedding(embeddingType=EmbeddingType.NOPRE_ALL)
            visitor_w2v_model = load_embedding(embeddingType=EmbeddingType.NOPRE_ALL)
        else:
            agent_w2v_model = load_embedding(embeddingType=EmbeddingType.NOPRE_AGENT)
            visitor_w2v_model = load_embedding(embeddingType=EmbeddingType.NOPRE_VISITOR)

    print('Pre-processing sequences')
    print(' - Get word vectors')
    agent_vocab_size, agent_embedding_dim, agent_word_indices, agent_embedding_matrix = \
        get_wordvectors_from_keyedvectors(agent_w2v_model, seed=seedNum)
    visitor_vocab_size, visitor_embedding_dim, visitor_word_indices, visitor_embedding_matrix = \
        get_wordvectors_from_keyedvectors(visitor_w2v_model, seed=seedNum)

    print(' - Transform sequences')
    transformed_agent_seq = transform_sequence(agent_sequence, word_indices=agent_word_indices)
    transformed_agent_time_seq = transform_time(agent_time_sequence, agent_maxtime, maxseq=time_maxseq)
    transformed_visitor_seq = transform_sequence(visitor_sequence, word_indices=visitor_word_indices)
    transformed_visitor_time_seq = transform_time(visitor_time_sequence, visitor_maxtime, maxseq=time_maxseq)

    print(' - Transform labels')
    transformed_labels = transform_label(label_data)
    print(' - Transform seq data to list')
    X_agent, X_visitor, X_agent_time, X_visitor_time, y = \
        transform_labeled_data_listform_separate(transformed_agent_seq, transformed_visitor_seq,
                                                 transformed_agent_time_seq, transformed_visitor_time_seq,
                                                 transformed_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seedNum)
    for train_index, test_index in sss.split(X_agent, y):
        pass

    X_agent_train,X_visitor_train,X_agent_time_train,X_visitor_time_train \
        = [X_agent[i] for i in train_index],[X_visitor[i] for i in train_index],\
          [X_agent_time[i] for i in train_index],[X_visitor_time[i] for i in train_index]
    X_agent_test, X_visitor_test, X_agent_time_test, X_visitor_time_test \
        = [X_agent[i] for i in test_index], [X_visitor[i] for i in test_index], \
          [X_agent_time[i] for i in test_index], [X_visitor_time[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]



    '''
    sss_for_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seedNum)
    for real_train_index, validation_index in sss_for_validation.split(X_agent_train, y_train):
        pass

    X_agent_realtrain, X_visitor_realtrain, X_agent_time_realtrain, X_visitor_time_realtrain \
        = [X_agent_train[i] for i in real_train_index], [X_visitor_train[i] for i in real_train_index], \
          [X_agent_time_train[i] for i in real_train_index], [X_visitor_time_train[i] for i in real_train_index]
    X_agent_val, X_visitor_val, X_agent_time_val, X_visitor_time_val \
        = [X_agent_train[i] for i in validation_index], [X_visitor_train[i] for i in validation_index], \
          [X_agent_time_train[i] for i in validation_index], [X_visitor_time_train[i] for i in validation_index]
    y_realtrain, y_validation = [y_train[i] for i in real_train_index], [y_train[i] for i in validation_index]

    X_agent_realtrain, X_visitor_realtrain, X_agent_time_realtrain, X_visitor_time_realtrain, y_realtrain = \
        random_oversampling_separate(X_agent_realtrain, X_visitor_realtrain,
                                     X_agent_time_realtrain, X_visitor_time_realtrain, y_realtrain, seed=seedNum)
    X_agent_test, X_visitor_test, X_agent_time_test, X_visitor_time_test, y_test = \
        random_oversampling_separate(X_agent_test, X_visitor_test, X_agent_time_test,
                                     X_visitor_time_test, y_test, seed=seedNum)
    '''

    X_agent_train, X_visitor_train, X_agent_time_train, X_visitor_time_train, y_train = \
        random_oversampling_separate(X_agent_train, X_visitor_train,
                                     X_agent_time_train, X_visitor_time_train, y_train, seed=seedNum)

    X_agent_test, X_visitor_test, X_agent_time_test, X_visitor_time_test, y_test = \
        random_oversampling_separate(X_agent_test, X_visitor_test,
                                     X_agent_time_test, X_visitor_time_test, y_test, seed=seedNum)


    X_agent_train = sequence.pad_sequences(X_agent_train, maxlen=maxseq)
    X_visitor_train = sequence.pad_sequences(X_visitor_train, maxlen=maxseq)
    X_agent_time_train = sequence.pad_sequences(X_agent_time_train,
                                                    dtype='float32', value=0., maxlen=time_maxseq)
    X_visitor_time_train = sequence.pad_sequences(X_visitor_time_train,
                                                      dtype='float32', value=0., maxlen=time_maxseq)

    X_agent_test = sequence.pad_sequences(X_agent_test, maxlen=maxseq)
    X_visitor_test = sequence.pad_sequences(X_visitor_test, maxlen=maxseq)
    X_agent_time_test = sequence.pad_sequences(X_agent_time_test, dtype='float32', value=0., maxlen=time_maxseq)
    X_visitor_time_test = sequence.pad_sequences(X_visitor_time_test, dtype='float32', value=0., maxlen=time_maxseq)

    print('X_agent_train shape:', X_agent_train.shape)
    print('X_visitor_train shape:', X_visitor_train.shape)
    print('X_agent_time_train shape:', X_agent_time_train.shape)
    print('X_visitor_time_train shape:', X_visitor_time_train.shape)

    print('X_agent_test shape:', X_agent_test.shape)
    print('X_visitor_test shape:', X_visitor_test.shape)
    print('X_agent_time_test shape:', X_agent_time_test.shape)
    print('X_visitor_time_test shape:', X_visitor_time_test.shape)

    print('Train...')

    list_callbacks = [CSVLogger(log_csvfile, separator=',', append=False)]
    if earlyStop:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        list_callbacks.append(earlyStopping)

    if modelType is ModelType.GRU_combined:
        model = GRU_combined(agent_vocab_size=agent_vocab_size, visitor_vocab_size=visitor_vocab_size, maxlen=maxseq,
                             dropout=dropout,
                             agent_embedding=agent_embedding_matrix, visitor_embedding=visitor_embedding_matrix,
                             agent_embedding_dim=agent_embedding_dim, visitor_embedding_dim=visitor_embedding_dim)()
        model.fit({'agent_text': X_agent_train, 'visitor_text': X_visitor_train,
                   'agent_time': X_agent_time_train, 'visitor_time': X_visitor_time_train}, y_train,
                  validation_data=({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                    'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test},
                               batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_combined:
        model = LSTM_combined(agent_vocab_size=agent_vocab_size, visitor_vocab_size=visitor_vocab_size, maxlen=maxseq,
                              time_maxlen=time_maxseq, dropout=dropout,
                              agent_embedding=agent_embedding_matrix, visitor_embedding=visitor_embedding_matrix,
                              agent_embedding_dim=agent_embedding_dim, visitor_embedding_dim=visitor_embedding_dim)()
        model.fit({'agent_text': X_agent_train, 'visitor_text': X_visitor_train,
                   'agent_time': X_agent_time_train, 'visitor_time': X_visitor_time_train}, y_train,
                  validation_data=({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                    'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test},
                               batch_size=batchSize, verbose=1)
    elif modelType is ModelType.RNN_combined:
        model = SimpleRNN_separate(agent_vocab_size=agent_vocab_size, visitor_vocab_size=visitor_vocab_size,
                                   maxlen=maxseq, time_maxlen=time_maxseq, dropout=dropout,
                                   agent_embedding=agent_embedding_matrix, visitor_embedding=visitor_embedding_matrix,
                                   agent_embedding_dim=agent_embedding_dim, visitor_embedding_dim=visitor_embedding_dim)()
        model.fit({'agent_text': X_agent_train, 'visitor_text': X_visitor_train,
                   'agent_time': X_agent_time_train, 'visitor_time': X_visitor_time_train}, y_train,
                  validation_data=({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                    'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test},
                               batch_size=batchSize, verbose=1)
    elif modelType is ModelType.CNN_combined:
        model = TextCNN_combined(agent_vocab_size=agent_vocab_size, visitor_vocab_size=visitor_vocab_size,
                                 maxlen=maxseq, time_maxlen=time_maxseq, dropout=dropout,
                                 agent_embedding=agent_embedding_matrix, visitor_embedding=visitor_embedding_matrix,
                                 agent_embedding_dim=agent_embedding_dim, visitor_embedding_dim=visitor_embedding_dim)()
        model.fit({'agent_text': X_agent_train, 'visitor_text': X_visitor_train,
                   'agent_time': X_agent_time_train, 'visitor_time': X_visitor_time_train}, y_train,
                  validation_data=({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                    'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'agent_text': X_agent_test, 'visitor_text': X_visitor_test,
                                'agent_time': X_agent_time_test, 'visitor_time': X_visitor_time_test},
                               batch_size=batchSize, verbose=1)
    elif modelType is ModelType.GRU_agent:
        model = GRU_single(vocab_size=agent_vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=agent_embedding_matrix, embedding_dim=agent_embedding_dim)()
        model.fit(X_agent_train, y_train,
                  validation_data=(X_agent_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_agent:
        model = LSTM_single(vocab_size=agent_vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=agent_embedding_matrix, embedding_dim=agent_embedding_dim)()
        model.fit(X_agent_train, y_train,
                  validation_data=(X_agent_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.RNN_agent:
        model = SimpleRNN_single(vocab_size=agent_vocab_size, maxlen=maxseq, dropout=dropout,
                                 embedding=agent_embedding_matrix, embedding_dim=agent_embedding_dim)()
        model.fit(X_agent_train, y_train,
                  validation_data=(X_agent_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.CNN_agent:
        model = TextCNN_single(vocab_size=agent_vocab_size, maxlen=maxseq, dropout=dropout,
                               embedding=agent_embedding_matrix, embedding_dim=agent_embedding_dim)()
        model.fit(X_agent_train, y_train,
                  validation_data=(X_agent_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.GRU_visitor:
        model = GRU_single(vocab_size=visitor_vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=visitor_embedding_matrix, embedding_dim=visitor_embedding_dim)()
        model.fit(X_visitor_train, y_train,
                  validation_data=(X_visitor_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_visitor:
        model = LSTM_single(vocab_size=visitor_vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=visitor_embedding_matrix, embedding_dim=visitor_embedding_dim)()
        model.fit(X_visitor_train, y_train,
                  validation_data=(X_visitor_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.RNN_visitor:
        model = SimpleRNN_single(vocab_size=visitor_vocab_size, maxlen=maxseq, dropout=dropout,
                                 embedding=visitor_embedding_matrix, embedding_dim=visitor_embedding_dim)()
        model.fit(X_visitor_train, y_train,
                  validation_data=(X_visitor_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.CNN_visitor:
        model = TextCNN_single(vocab_size=visitor_vocab_size, maxlen=maxseq, dropout=dropout,
                               embedding=visitor_embedding_matrix, embedding_dim=visitor_embedding_dim)()
        model.fit(X_visitor_train, y_train,
                  validation_data=(X_visitor_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.GRU_agenttime:
        model = GRU_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_agent_time_train, y_train,
                  validation_data=(X_agent_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_agenttime:
        model = LSTM_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_agent_time_train, y_train,
                  validation_data=(X_agent_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.RNN_agenttime:
        model = SimpleRNN_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_agent_time_train, y_train,
                  validation_data=(X_agent_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.CNN_agenttime:
        model = TextCNN_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_agent_time_train, y_train,
                  validation_data=(X_agent_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_agent_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.GRU_visitortime:
        model = GRU_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_visitor_time_train, y_train,
                  validation_data=(X_visitor_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_visitortime:
        model = LSTM_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_visitor_time_train, y_train,
                  validation_data=(X_visitor_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.RNN_visitortime:
        model = SimpleRNN_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_visitor_time_train, y_train,
                  validation_data=(X_visitor_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_time_test, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.CNN_visitortime:
        model = TextCNN_single_time(maxlen=time_maxseq, dropout=dropout)()
        model.fit(X_visitor_time_train, y_train,
                  validation_data=(X_visitor_time_test, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict(X_visitor_time_test, batch_size=batchSize, verbose=1)

    else:
        print('Model type is not specified in an expected way.')
        exit()

    print('Evaluation..')
    with open(result_file, 'wt') as f:
        writer.eval(y_pred, y_test, file=f)
