from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def word2features(sent, i, snow):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'stemming': snow.stem(word.decode("utf8")),
        'is_med': word in medical_dict,
        # "headofnoun": False,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
        # if postag[:2] == "NN" and postag1[:2]!= "NN":
        #     features.update({
        #         "headofnoun": True
        #     })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent, snow):
    return [word2features(sent, i, snow) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def word2features_nopost(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'stemming':  stem.stem(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features_nopost(sent):
    return [word2features_nopost(sent, i) for i in range(len(sent))]


def sent2labels_nopost(sent):
    return [label for token, label in sent]


def sent2tokens_nopost(sent):
    return [token for token, label in sent]


def load_data_nopost(path):
    f = open(path, 'r')
    all = f.read().split('\n\n')
    all_data = []
    for sent in all:
        if sent == "":
            continue
        sent_list = []
        sent_data = sent.split('\n')
        for s in sent_data:
            s_data = s.split()
            sent_list.append((s_data[0], s_data[1]))
        all_data.append(sent_list)
    return all_data


def load_data(path):
    f = open(path, 'r')
    all = f.read().split('\n\n')
    all_data = []
    for sent in all:
        if sent == "":
            continue
        sent_list = []
        origin_sent = []
        sent_data = sent.split('\n')
        for s in sent_data:
            s_data = s.split()
            origin_sent.append(s_data[0])
        nltk_pos_tag = nltk.pos_tag(origin_sent)
        for i in range(len(sent_data)):
            s_data = sent_data[i].split()
            sent_list.append((s_data[0], nltk_pos_tag[i][1], s_data[1]))
        all_data.append(sent_list)
    return all_data


def generate_result(path, predict):
    f = open(path, 'r')
    out = open("result", 'w')
    pre_data = []
    for sent_pre in predict:
        for p in sent_pre:
            pre_data.append(p)
    line = f.readline()
    idx = 0
    while line:
        if line.strip() != '':
            write_line = line.replace('\n', '') + " " + pre_data[idx]
            idx += 1
            out.write(write_line + '\n')
        else:
            out.write(line)
        line = f.readline()
    out.close()


def load_test_data(path):
    f = open(path, 'r')
    all = f.read().split('\n\n')
    all_data = []
    for sent in all:
        if sent == "":
            continue
        sent_list = []
        origin_sent = []
        sent_data = sent.split('\n')
        for s in sent_data:
            origin_sent.append(s)
        nltk_pos_tag = nltk.pos_tag(origin_sent)
        for i in range(len(sent_data)):
            s_data = sent_data[i]
            sent_list.append((s_data, nltk_pos_tag[i][1]))
        all_data.append(sent_list)
    return all_data


def load_med_dict(path):
    f = open(path, 'r')
    all = f.read().split('\n')
    med_dict = {}
    for m in all:
        med_dict[m] = 0
    return med_dict


if __name__ == "__main__":
    # train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    # test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    # test_sents = load_test_data("data/test.eval")
    stem = nltk.stem.SnowballStemmer('english')
    labels = ['B-problem', 'I-problem', 'B-test', 'I-test', 'B-treatment', 'I-treatment']
    medical_dict = load_med_dict("wordlist.txt")
    # train_sents = load_data_nopost("data/train.eval")
    # test_sents = load_data_nopost("data/dev.eval")
    # X_train = [sent2features_nopost(s) for s in train_sents]
    # y_train = [sent2labels_nopost(s) for s in train_sents]
    #
    # X_test = [sent2features_nopost(s) for s in test_sents]
    # y_test = [sent2labels_nopost(s) for s in test_sents]
    train_sents = load_data("data/train.eval")
    test_sents = load_data("data/dev.eval")
    X_train = [sent2features(s, stem) for s in train_sents]
    X_test = [sent2features(s, stem) for s in test_sents]
    y_train = [sent2labels(s) for s in train_sents]
    y_test = [sent2labels(s) for s in test_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        # 'c1': [0,0,1,0,2,0,3,0,4,0.5,0.6,0.7,0.8,0,9,1],
        # 'c2': [0,0,1,0,2,0,3,0,4,0.5,0.6,0.7,0.8,0,9,1],
        'c1': [i/100.0 for i in range(3, 6)],
        'c2': [i/100.0 for i in range(3, 6)],
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    gs = GridSearchCV(crf, params_space, cv=3, verbose=1, n_jobs=-1, scoring=f1_scorer)
    gs.fit(X_train, y_train)
    print('best params:', gs.best_params_)
    print('best CV score:', gs.best_score_)
    print('model size: {:0.2f}M'.format(gs.best_estimator_.size_ / 1000000))

    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     # c1=0.08624993064986249,
    #     # c2=0.06097519251873882,
    #     # nopost
    #     # c1=0.009868212760639313,
    #     # c2=0.04134003944483053,
    #     # post
    #     c1=0.08,
    #     c2=0.05,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # crf.fit(X_train, y_train)
    # y_pred = crf.predict(X_test)
    # metrics.flat_f1_score(y_test, y_pred,
    #                       average='weighted', labels=labels)
    # sorted_labels = sorted(
    #     labels,
    #     key=lambda name: (name[1:], name[0])
    # )
    # print(metrics.flat_classification_report(
    #     y_test, y_pred, labels=sorted_labels, digits=3
    # ))
    # generate_result("data/dev.eval", y_pred)
