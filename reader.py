# -*- encoding: utf-8 -*-
import os, codecs
import preprocessing
from utils import *


BASE_PATH = 'samsungchat'
AGENT_NOPRE_PATH = os.path.join(BASE_PATH, 'agent_sequence.txt')
VISITOR_NOPRE_PATH = os.path.join(BASE_PATH, 'visitor_sequence.txt')
AGENT_PRE_PATH = os.path.join(BASE_PATH, 'agent_stemmed_sequence.txt')
VISITOR_PRE_PATH = os.path.join(BASE_PATH, 'visitor_stemmed_sequence.txt')

LABEL_PATH = os.path.join(BASE_PATH, 'session_satisfaction.txt')


def load_data(preprocess, maxseq, encodeTime):

    session_data = {}
    if preprocessing:
        agent_path = AGENT_PRE_PATH
        visitor_path = VISITOR_PRE_PATH
    else:
        agent_path = AGENT_NOPRE_PATH
        visitor_path = VISITOR_PRE_PATH

    fvisitor = codecs.open(visitor_path, 'r', encoding='utf-8')
    fagent = codecs.open(agent_path, 'r', encoding='utf-8')

    visitor_lines = fvisitor.readlines()
    agent_lines = fagent.readlines()
    for ix, v_line in enumerate(visitor_lines):

        a_line = agent_lines[ix]
        v_data = v_line.strip().split()
        a_data = a_line.strip().split()
        v_session_id = v_data[0]
        a_session_id = a_data[0]

        if a_session_id != v_session_id:
            print('There are unknown errors on reading lines. line number:{}'.format(ix))
            exit()
        else:
            session_id = a_session_id

        v_target = v_data[1:]
        a_target = a_data[1:]
        words = []

        a_target_len = len(a_target)

        for wix, v_w in enumerate(v_target):
            if wix >= a_target_len:
                print('Unexpected error.')
                print(len(a_data))
                print(len(v_data))
                exit()
            a_w = a_target[wix]
            if a_w == v_w:
                # time indicator
                if encodeTime is True:
                    words.append(preprocessing.NULL_TOKEN)
                elif encodeTime is False:
                    pass

            elif a_w == preprocessing.NULL_TOKEN:
                words.append(v_w)
            elif v_w == preprocessing.NULL_TOKEN:
                words.append(a_w)

        session_data[session_id] = words[:maxseq]
    fvisitor.close()
    fagent.close()

    return session_data


def load_data_separate(preprocess, maxseq):

    agent_text, visitor_text, agent_time, visitor_time = {},{},{},{}
    if preprocess:
        agent_path = AGENT_PRE_PATH
        visitor_path = VISITOR_PRE_PATH
    else:
        agent_path = AGENT_NOPRE_PATH
        visitor_path = VISITOR_NOPRE_PATH

    times_for_max = []
    with codecs.open(agent_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            session_id = data[0]
            words,times = [],[]

            total_cnt = 0
            for i, w in enumerate(data[1:]):

                if 'AGENT_' in w:
                    time = float(w[6:])
                    if time < 0:
                        time = 0
                    if total_cnt < maxseq:
                        times.append(time)
                    times_for_max.append(time)
                elif 'VISITOR_' in w:
                    pass
                elif w == preprocessing.NULL_TOKEN:
                    total_cnt += 1
                else:
                    total_cnt += 1
                    if total_cnt < maxseq:
                        words.append(w)

            agent_text[session_id] = words
            agent_time[session_id] = times
    agent_maxtime = max(times_for_max)

    times_for_max = []
    with codecs.open(visitor_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            session_id = data[0]
            words,times = [],[]
            times_for_max = []

            total_cnt = 0
            for i, w in enumerate(data[1:]):

                if 'AGENT_' in w:
                    pass
                elif 'VISITOR_' in w:
                    time = float(w[8:])
                    if time < 0:
                        time = 0
                    if total_cnt < maxseq:
                        times.append(time)
                    times_for_max.append(time)
                elif w == preprocessing.NULL_TOKEN:
                    total_cnt += 1
                else:
                    total_cnt += 1
                    if total_cnt < maxseq:
                        words.append(w)

            visitor_text[session_id] = words
            visitor_time[session_id] = times
    visitor_maxtime = max(times_for_max)

    return agent_text,visitor_text,agent_time,visitor_time,agent_maxtime,visitor_maxtime


def load_label():
    label_for_session = {}
    with open(LABEL_PATH, 'rt') as f:

        f.next()
        for line in f:

            data = line.strip().split('|')
            session_id = data[0]
            label = data[1].strip()
            label_for_session[session_id] = label

    return label_for_session


def compare_data_agent_visitor():
    #paths = [(VISITOR_STEMMED_SESSION_PATH, AGENT_STEMMED_SESSION_PATH), (VISITOR_SESSION_PATH, AGENT_SESSION_PATH)]
    paths = [(VISITOR_NOPRE_PATH, AGENT_NOPRE_PATH)]
    for (visitor_path, agent_path) in paths:
        fvisitor = codecs.open(visitor_path, 'r', encoding='utf-8')
        fagent = codecs.open(agent_path, 'r', encoding='utf-8')

        visitor_lines = fvisitor.readlines()
        agent_lines = fagent.readlines()
        for ix, v_line in enumerate(visitor_lines):

            a_line = agent_lines[ix]
            v_data = v_line.strip().split(' ')
            a_data = a_line.strip().split(' ')
            v_session_id = v_data[0]
            a_session_id = a_data[0]

            if v_session_id != a_session_id:
                print('Discrepancy on session ids')
                print(v_session_id)
                exit()

            v_target = v_data[1:]
            a_target = a_data[1:]

            if len(v_target) != len(a_target):

                print('Discrepancy on sequence lengths')
                print('Linenum:{}'.format(ix))
                print('Session id:{}'.format(v_session_id))
                print(len(v_target))
                print(len(a_target))
                exit()

        fvisitor.close()
        fagent.close()


if __name__ == '__main__':
    compare_data_agent_visitor()
