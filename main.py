# -*- encoding: utf-8 -*-
from utils import EmbeddingType, DataType, ModelType
from trainer import read_train_eval


if __name__ == '__main__':

    #read_train_eval(dataType=DataType.ALL, allDataForEmbed=True, preprocess=True, encodeTime=True, maxseq=200,
    #                modelType=ModelType.GRU_visitor, dropout=0.1, earlyStop=True, seedNum=20160430, batchSize=70,
    #                maxEpoch=100)
    read_train_eval(testid="agent", allDataForEmbed=True, preprocess=True, encodeTime=True, maxseq=200,
                    modelType=ModelType.GRU_agent, dropout=0.1, earlyStop=False, seedNum=20160430, batchSize=70,
                    maxEpoch=100)
