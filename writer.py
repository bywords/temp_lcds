# -*- encoding:utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def eval(pred, label, file=sys.stdout):
    y_pred = np.reshape(pred, len(pred))
    y_predict = []
    for x in y_pred:
        if x >= 0.5:
            y_predict.append(1)
        else:
            y_predict.append(0)

    #y_predict = [arr[0] for arr in pred]
    print('f1,{}'.format(f1_score(label, y_predict)), file=file)
    print('precision,{}'.format(precision_score(label, y_predict)), file=file)
    print('recall,{}'.format(recall_score(label, y_predict)), file=file)
    print('accuracy,{}'.format(accuracy_score(label, y_predict)), file=file)
