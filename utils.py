# -*- encoding:utf-8 -*-


class EmbeddingType:

    NOPRE_AGENT = 0
    NOPRE_VISITOR = 1
    NOPRE_ALL = 2
    PRE_AGENT = 3
    PRE_VISITOR = 4
    PRE_ALL = 5
    GLOVE_6B_50D = 6
    GLOVE_6B_100D = 7
    GLOVE_6B_200D = 8
    GLOVE_6B_300D = 9


class DataType:

    ALL = 1
    AGENT = 2
    VISITOR = 3


class ModelType:

    RNN_combined = 1
    GRU_combined = 2
    LSTM_combined = 3
    CNN_combined = 4
    RNN_single = 17
    GRU_single = 18
    LSTM_single = 19
    CNN_single = 20
    RNN_agent = 5
    GRU_agent = 6
    LSTM_agent = 7
    CNN_agent = 8
    RNN_visitor = 9
    GRU_visitor = 10
    LSTM_visitor = 11
    CNN_visitor = 12
    RNN_agenttime = 13
    GRU_agenttime = 14
    LSTM_agenttime = 15
    CNN_agenttime = 16
    RNN_visitortime = 13
    GRU_visitortime = 14
    LSTM_visitortime = 15
    CNN_visitortime = 16

