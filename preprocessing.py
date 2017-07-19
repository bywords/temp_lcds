# -*- encoding:utf-8 -*-
from __future__ import print_function
import os, codecs, re
import random
import reader
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from nltk.stem.porter import PorterStemmer


BASE_DIR = 'samsungchat'
THREAD_PATH = os.path.join(BASE_DIR,'sequence_samsung.txt')
UTTERANCE_LENGTH_PATH = os.path.join(BASE_DIR,'utt_length.txt')

NULL_TOKEN = "X"


def encode_time_bucket(interval, speaker):

    if interval < 10:
        if speaker == "Agent":
            return "AGENT_TIME_1"
        else:
            return "VISITOR_TIME_1"
    elif interval < 60:
        if speaker == "Agent":
            return "AGENT_TIME_2"
        else:
            return "VISITOR_TIME_2"
    else:
        if speaker == "Agent":
            return "AGENT_TIME_3"
        else:
            return "VISITOR_TIME_3"


def encode_time(time1, time2, speaker):

    interval=(time2 - time1).total_seconds()

    if speaker == "Agent":
        return "AGENT_{}".format(interval)
    else:
        return "VISITOR_{}".format(interval)


def load_target_sessions(threshold):

    s_set=[]
    with open(UTTERANCE_LENGTH_PATH, 'rt') as f:
        for line in f:
            data=line.strip().split()
            sid=data[0]
            ulen=int(data[1])

            if ulen < threshold:
                continue
            s_set.append(sid)
    return s_set


def make_session_text():

    session_set = load_target_sessions(threshold=4)
    fsession_agent = codecs.open(reader.AGENT_SESSION_PATH, 'w', encoding='utf-8')
    fsession_visitor = codecs.open(reader.VISITOR_SESSION_PATH, 'w', encoding='utf-8')

    with codecs.open(THREAD_PATH, 'r', encoding='utf-8') as f:
        f.next()
        current_id = 'EXID'
        prev_utt_dt = None
        prev_speaker = None
        agent_text_list = []
        visitor_text_list = []

        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                print("Reading {}th line".format(idx))
            data = line.strip().split('|')
            session_id = data[0]
            if session_id not in session_set or len(data) != 6:
                continue
            speaker = data[2]
            type = data[4]

            # 2013-10-01T00:02:47+00:00
            current_utt_dt = datetime.strptime(data[5][:19], '%Y-%m-%dT%H:%M:%S')
            if session_id != current_id and current_id != 'EXID':
                current_id = session_id
                num_agent_words = len(agent_text_list)
                num_visitor_words = len(visitor_text_list)
                # save agent sessions
                print(current_id, file=fsession_agent, end=' ')
                for ix in range(num_agent_words-1):
                    print(agent_text_list[ix], file=fsession_agent, end=' ')
                print(agent_text_list[-1], file=fsession_agent)
                agent_text_list[:] = []
                # save visitor sessions
                print(current_id, file=fsession_visitor, end=' ')
                for ix in range(num_visitor_words-1):
                    print(visitor_text_list[ix], file=fsession_visitor, end=' ')
                print(visitor_text_list[-1], file=fsession_visitor)
                visitor_text_list[:] = []
                prev_utt_dt = None

            elif session_id != current_id and current_id == 'EXID':
                current_id = session_id
            elif session_id == current_id:
                pass
            else:
                print("Unexpected errors on reading threads")
                exit()

            if type == 'URL':
                text = 'URLLINK'
                tokens = [text]
            else:

                text = data[3].strip().lower()
                text = re.sub(r'<a href>.*<\a>', 'HTMLLINK', text)
                text = re.sub(r'http://[\w./]+', 'URLLINK', text)
                tokens = [re.sub(r'\W+', '', t) for t in re.split(r'\s+', text)]
                tokens = [t for t in tokens if t != ""]

            if prev_utt_dt is not None and prev_speaker != speaker:
                time_bucket = encode_time(prev_utt_dt, current_utt_dt, speaker)
                agent_text_list.append(time_bucket)
                visitor_text_list.append(time_bucket)

            if speaker == "Agent":
                for t in tokens:
                    agent_text_list.append(t)
                    visitor_text_list.append(NULL_TOKEN)
            elif speaker == "Visitor":
                for t in tokens:
                    agent_text_list.append(NULL_TOKEN)
                    visitor_text_list.append(t)
            else:
                print("Probably errors on data. line num:{}".format(idx))
                exit()

            prev_speaker = speaker
            prev_utt_dt = current_utt_dt

        # For the last session
        num_agent_words = len(agent_text_list)
        num_visitor_words = len(visitor_text_list)
        # save agent sessions
        print(current_id, file=fsession_agent, end=' ')
        for ix in range(num_agent_words - 1):
            print(agent_text_list[ix], file=fsession_agent, end=' ')
        print(agent_text_list[-1], file=fsession_agent)
        agent_text_list[:] = []
        # save visitor sessions
        print(current_id, file=fsession_visitor, end=' ')
        for ix in range(num_visitor_words - 1):
            print(visitor_text_list[ix], file=fsession_visitor, end=' ')
        print(visitor_text_list[-1], file=fsession_visitor)
        visitor_text_list[:] = []

    fsession_agent.close()
    fsession_visitor.close()


def make_stemmed_text():
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    print('Stemming agent sessions.')
    fout_agent = codecs.open(reader.AGENT_STEMMED_SESSION_PATH, 'w', encoding='utf-8')
    with codecs.open(reader.AGENT_SESSION_PATH, 'r', encoding='utf-8') as fsession_agent:
        for idx, line in enumerate(fsession_agent):
            if idx % 1000 == 0:
                print("Reading {}th line".format(idx))

            data = line.strip().split()
            session_id = data[0]
            words = data[1:]
            num_words = len(words)
            print(session_id, file=fout_agent, end=' ')

            for ix in range(num_words-1):
                w = words[ix]
                if 'AGENT_' in w or 'VISITOR_' in w:
                    print(w, file=fout_agent, end=' ')
                else:
                    print(p_stemmer.stem(w), file=fout_agent, end=' ')

            print(p_stemmer.stem(words[-1]), file=fout_agent)

    fout_agent.close()
    print('Stemming visitor sessions.')
    fout_visitor = codecs.open(reader.VISITOR_STEMMED_SESSION_PATH, 'w', encoding='utf-8')
    with codecs.open(reader.VISITOR_SESSION_PATH, 'r', encoding='utf-8') as fsession_visitor:
        for idx, line in enumerate(fsession_visitor):
            if idx % 1000 == 0:
                print("Reading {}th line".format(idx))

            data = line.strip().split()
            session_id = data[0]
            words = data[1:]
            num_words = len(words)
            print(session_id, file=fout_visitor, end=' ')

            for ix in range(num_words - 1):
                w = words[ix]
                if 'AGENT_' in w or 'VISITOR_' in w:
                    print(w, file=fout_visitor, end=' ')
                else:
                    print(p_stemmer.stem(w), file=fout_visitor, end=' ')

            print(p_stemmer.stem(words[-1]), file=fout_visitor)
    fout_visitor.close()


def transform_label(session_label):
    for sid, label in session_label.items():
        if label == "Very  Dissatisfied" or label == "Dissatisfied":
            # 1 if satisfaction shows negativity
            label_int = 1
        else:
            label_int = 0
        session_label[sid] = label_int

    return session_label


def transform_sequence(sequences, word_indices):

    new_sequences = defaultdict(list)
    for sid, sequence in sequences.items():
        for s_ix, w in enumerate(sequence):

            if w not in word_indices:
                index = word_indices[NULL_TOKEN]
            else:
                index = word_indices[w]
            new_sequences[sid].append(index)

    return new_sequences


def transform_time(sequences, maxtime, maxseq):

    max_log_time = np.log10(maxtime+1)

    new_sequences = {}
    for sid, time_seq in sequences.items():
        log_time_temp = np.log10(np.asarray(time_seq, dtype=np.float32)+1.0)
        normalized_time = list(log_time_temp/max_log_time)
        time_sequence = [np.asarray([t], dtype=np.float32) for t in normalized_time[:maxseq]]

        #num_left_step = maxseq - len(time_sequence)
        #time_sequence.extend([np.asarray([.0], dtype=np.float32)]*num_left_step)

        new_sequences[sid] = time_sequence

    return new_sequences


def transform_labeled_data_listform(sequences, labels):

    sids = sequences.keys()
    X, y = [], []
    for sid in sids:
        seq = sequences[sid]
        label = labels[sid]
        X.append(seq)
        y.append(label)
    return X, y


def transform_labeled_data_listform_separate(agent_seq, visitor_seq, agent_time_seq, visitor_time_seq, labels):

    sids = agent_seq.keys()
    X_aseq, X_vseq, X_atseq, X_vtseq, y = [],[],[],[],[]
    for sid in sids:
        aseq = agent_seq[sid]
        vseq = visitor_seq[sid]
        atseq = agent_time_seq[sid]
        vtseq = visitor_time_seq[sid]
        label = labels[sid]
        X_aseq.append(aseq)
        X_vseq.append(vseq)
        X_atseq.append(atseq)
        X_vtseq.append(vtseq)
        y.append(label)
    return X_aseq, X_vseq, X_atseq, X_vtseq, y



def filter_labeled_data(sequences, labels):
    seq_keys = set(sequences.keys())
    label_keys = set(labels.keys())
    common_keys = seq_keys & label_keys
    new_sequences = {}
    new_labels = {}

    for k in common_keys:
        new_sequences[k] = sequences[k]
        new_labels[k] = labels[k]

    return new_sequences, new_labels


def filter_labeled_data_separate(agent_sequence, visitor_sequence, agent_time_sequence,
                                 visitor_time_sequence, label_data):
    agent_seq_keys = set(agent_sequence.keys())
    visitor_seq_keys = set(visitor_sequence.keys())
    agent_time_seq_keys = set(agent_time_sequence.keys())
    visitor_time_seq_keys = set(visitor_time_sequence.keys())
    label_keys = set(label_data.keys())
    common_keys = agent_seq_keys & visitor_seq_keys & agent_time_seq_keys & visitor_time_seq_keys & label_keys

    new_agent_seq = {}
    new_visitor_seq = {}
    new_agent_time_seq = {}
    new_visitor_time_seq = {}
    new_labels = {}

    for k in common_keys:
        new_agent_seq[k] = agent_sequence[k]
        new_visitor_seq[k] = visitor_sequence[k]
        new_agent_time_seq[k] = agent_time_sequence[k]
        new_visitor_time_seq[k] = visitor_time_sequence[k]
        new_labels[k] = label_data[k]

    return new_agent_seq, new_visitor_seq, new_agent_time_seq, new_visitor_time_seq, new_labels



def random_oversampling(X, y, seed):
    random.seed(seed)
    satisfaction_counter = Counter(y)
    (major_label, major_cnt), (minor_label, minor_cnt) = satisfaction_counter.most_common(2)
    minor_indices = [i for i, x in enumerate(y) if x == minor_label]
    all_indices = [i for i in range(0, len(y))]
    for i in range(major_cnt / minor_cnt - 2):
        all_indices.extend(minor_indices)
    rest_cnt = major_cnt % minor_cnt
    all_indices.extend(random.sample(minor_indices, rest_cnt))

    new_X = [X[i] for i in all_indices]
    new_y = [y[i] for i in all_indices]

    return new_X, new_y


def random_oversampling_separate(X_agent, X_visitor, X_agent_time, X_visitor_time, y, seed):
    random.seed(seed)
    satisfaction_counter = Counter(y)
    (major_label, major_cnt), (minor_label, minor_cnt) = satisfaction_counter.most_common(2)
    minor_indices = [i for i, x in enumerate(y) if x == minor_label]
    all_indices = [i for i in range(0, len(y))]
    for i in range(major_cnt / minor_cnt - 2):
        all_indices.extend(minor_indices)
    rest_cnt = major_cnt % minor_cnt
    all_indices.extend(random.sample(minor_indices, rest_cnt))

    new_X_agent = [X_agent[i] for i in all_indices]
    new_X_visitor = [X_visitor[i] for i in all_indices]
    new_X_agent_time = [X_agent_time[i] for i in all_indices]
    new_X_visitor_time = [X_visitor_time[i] for i in all_indices]
    new_y = [y[i] for i in all_indices]

    return new_X_agent, new_X_visitor, new_X_agent_time, new_X_visitor_time, new_y


def get_wordvectors_from_keyedvectors(keyedVectors, seed):

    OOV_TOKEN = 'OOV_TOKEN'
    vocab_list_wo_NULL = keyedVectors.vocab.keys()
    embedding_dim = keyedVectors.syn0.shape[1]

    if NULL_TOKEN in vocab_list_wo_NULL:
        vocab_list_wo_NULL.remove(NULL_TOKEN)
    vocab_list = [OOV_TOKEN, NULL_TOKEN] + vocab_list_wo_NULL # X should be located at 0

    word_indices = {}
    for vocab_index in range(2, len(vocab_list)):
        w = vocab_list[vocab_index]
        word_indices[w] = vocab_index

    # index for out-of-vocabulary words
    np.random.seed(seed)
    word_indices[OOV_TOKEN] = 0
    word_indices[NULL_TOKEN] = 1
    vocab_size = len(word_indices)

    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    # Out-of-vocabulary word is zero-vector
    embedding_matrix[1] = np.random.random(size=keyedVectors.syn0.shape[1]) / 5 - 0.1
    for word in vocab_list_wo_NULL:
        i = word_indices[word]
        embedding_vector = keyedVectors.word_vec(word)
        embedding_matrix[i] = embedding_vector

    return vocab_size, embedding_dim, word_indices, embedding_matrix


if __name__ == '__main__':

    make_session_text()
    make_stemmed_text()
