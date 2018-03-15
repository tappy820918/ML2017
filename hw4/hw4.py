#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional ,Dropout
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import *
from gensim.models import word2vec, Word2Vec

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 260
VALIDATION_SPLIT = 0.1

batchsize = 3000
num_epoch = 62
Min_count = 5
####################
training_label = 'c:/Users/user/Desktop/training_label.txt'
training_nolabel = 'c:/Users/user/Desktop/training_nolabel.txt'
testing = 'c:/Users/user/Desktop/testing_data.txt'
toy = 'c:/Users/user/Desktop/toy.txt'

train_text_path = 'c:/Users/user/Desktop/feature/train_text.npy'
y_train_path = 'c:/Users/user/Desktop/feature/y_train.npy'
valid_text_path = 'c:/Users/user/Desktop/feature/valid_text.npy'
test_text_path = 'c:/Users/user/Desktop/feature/test_text.npy'

predict_path = 'c:/Users/user/Desktop/result/predict.csv'
filepath = "c:/Users/user/Desktop/weight/weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
good_incidence_path = 'c:/Users/user/Desktop/feature/good_incidence.npy'
W2Vmodel_path = 'C:/Users/user/Desktop/W2Vmodel_no_mark'


####################


def load_data(training_label, training_nolabel, testing):
    with open(training_label, "r", encoding='utf-8-sig') as f:
        y_train = []
        train_text = []
        test_text = []
        for l in f:
            y_train.append(l.strip().split("+++$+++")[0])
            train_text.append(l.strip().split("+++$+++")[1])
    valid_text = [line.strip() for line in open(
            training_nolabel, "r", encoding='utf-8-sig')]
    test_text = [line.strip().split(',', 1)[1] for line in open(
            testing, "r", encoding='utf-8-sig')][1::]
    return y_train, train_text, valid_text, test_text


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



def savemodel(model, path):
    model.save_weights(path)
    print("Saved model to disk")


def buildmodel():
    model = Sequential()
    model.add(Embedding(len(embeddings_matrix), EMBEDDING_DIM,
                        weights=[embeddings_matrix], trainable=False))
    # model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(LSTM(256))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(256)))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    model.summary()
    return model


def Set_Data(readnpy=True):
    if readnpy:
        y_train, train_text, valid_text, test_text = load_data(
                training_label, training_nolabel, testing)
        np.save(train_text_path, train_text)
        np.save(y_train_path, y_train)
        np.save(valid_text_path, valid_text)
        np.save(test_text_path, test_text)
    else:
        train_text = np.load(train_text_path)
        y_train = np.load(y_train_path)
        valid_text = np.load(valid_text_path)
        test_text = np.load(test_text_path)
    return train_text, y_train, valid_text, test_text


def Set_W2V_model():
    documents = np.concatenate([train_text, valid_text, test_text])
    sentences = [text_to_word_sequence(s, filters='', lower=True, split=" ") for s in
                 documents.tolist()]
    W2Vmodel = Word2Vec(sentences, sg=1, size=EMBEDDING_DIM,
                        window=5, min_count=Min_count, workers=8)
    W2Vmodel.save(W2Vmodel_path)
    
    return W2Vmodel


def write_result(prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(prediction)):
        s.writerow([i, prediction[i][0]])
    text.close()


def prob_to_one_hot(prob):
    if prob < 0.5:
        return ('0')
    else:
        return ('1')


def index_array(X, max_length, EMBEDDING_DIM):
    return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],
                           np.zeros((max_length - len(X)))])


def set_embedding_matrix(W2Vmodel_path):
    W2Vmodel = Word2Vec.load(W2Vmodel_path)
    word2idx = {"_PAD": 0}
    vocab_list = [(k, W2Vmodel.wv[k]) for k, v in W2Vmodel.wv.vocab.items()]
    embeddings_matrix = np.zeros(
            (len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    return embeddings_matrix

#  ×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
def set_training_vector():
    train_list = [text_to_word_sequence( s, filters='', lower=True, split=" ") for s in train_text]
    test_list = [text_to_word_sequence( s, filters='', lower=True, split=" ") for s in test_text]
    valid_list = [text_to_word_sequence(  s, filters='', lower=True, split=" ") for s in valid_text]
    x_train = np.array( [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in train_list])
    x_test = np.array( [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in test_list])
    x_valid = np.array([index_array(x[:(MAX_SEQUENCE_LENGTH - 1)], MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in valid_list])
    return x_train, x_test, x_valid


train_text, y_train, valid_text, test_text = Set_Data()
# W2Vmodel = Set_W2V_model()
W2Vmodel = Word2Vec.load(W2Vmodel_path)

print(W2Vmodel.most_similar(['go']))
print(W2Vmodel.most_similar(['damn']))
print(W2Vmodel.most_similar(['sooooon']))
#
# W2Vmodel = Word2Vec.load(W2Vmodel_path)
word2idx = {"_PAD": 0}
vocab_list = [(k, W2Vmodel.wv[k]) for k, v in W2Vmodel.wv.vocab.items()]
embeddings_matrix = np.zeros((len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]

x_train, x_test, x_valid = set_training_vector()

ACCearlyStopping = EarlyStopping(
        monitor='val_acc', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model = buildmodel()
train_history = model.fit(x_train, y_train, verbose=2,
                          batch_size=2000, epochs=80,
                          callbacks=[checkpoint, ACCearlyStopping], validation_split=VALIDATION_SPLIT,
                          class_weight='auto')

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# Predict model
model = load_model('c:/Users/user/Desktop/weight/weights-improvement-79-0.823.hdf5')
#model = load_model('c:/Users/user/Desktop/weights-improvement-57-0.819.hdf5')

prediction = model.predict_classes(x_test, batch_size=5000)

write_result(prediction)

##############################################################################
'''
    Start unsupervised learning step
'''
valid_predict = model.predict(x_valid, batch_size=10000)
good_incidence = []
good_incidence_y = []

for i in range(len(valid_predict)):
    if abs(valid_predict[i] - 0.5) > 0.47:
        good_incidence_y.append(prob_to_one_hot(valid_predict[i]))
        good_incidence.append(list(x_valid[i]))
        print(i)
np.save(good_incidence_path, good_incidence)
y_final = np.concatenate([good_incidence_y, y_train])
x_final = np.concatenate([good_incidence, x_train])

SemiSupervisedModel = buildmodel()
Finalpath = "c:/Users/user/Desktop/weight/Model-{epoch:02d}-{val_acc:.3f}.hdf5"
Finalcheckpoint = ModelCheckpoint(Finalpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

train_history = SemiSupervisedModel.fit(x_final, y_final, verbose=2,
                                        batch_size=batchsize, epochs=40,
                                        callbacks=[Finalcheckpoint, ACCearlyStopping], validation_split=0.1,
                                        class_weight='auto')

Finalpath = "c:/Users/user/Desktop/weight/Model-03-0.811.hdf5"
SemiSupervisedModel = load_model("c:/Users/user/Desktop/weight/Model-01-0.804.hdf5")
prediction = SemiSupervisedModel.predict_classes(x_test, batch_size=10000)
write_result(prediction, 'c:/Users/user/Desktop/result/Semi_predict.csv')

##############################################################################
'''
    Start Bag of Words learning step
'''

from keras.preprocessing.text import Tokenizer

documents = np.concatenate([train_text, valid_text, test_text])



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(documents)
print(tokenizer.word_index)
tokenizer.texts_to_sequences(train_text)
x_train_BoW = tokenizer.texts_to_matrix(train_text,mode='count')
tokenizer.texts_to_sequences(test_text)
x_test_BoW = tokenizer.texts_to_matrix(test_text,mode='count')

BOW_model = Sequential()
BOW_model.add(Dense(input_dim=x_train_BoW.shape[1],units =1024,activation='relu',kernel_regularizer=regularizers.l1(0.001)))
BOW_model.add(Dense(256, activation='relu'))
BOW_model.add(Dropout(0.5))
BOW_model.add(Dense(units=1, activation='relu'))
BOW_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-3), metrics=['accuracy'])
BOW_model.summary()

BOW_train_history = BOW_model.fit(x_train_BoW, y_train, verbose=2,batch_size=7000, epochs=40,validation_split=0.1,class_weight='auto')
show_train_history(BOW_train_history, 'acc', 'val_acc')
show_train_history(BOW_train_history, 'loss', 'val_loss')

BOW_prediction = BOW_model.predict_classes(x_test_BoW, batch_size=5000)

write_result(BOW_prediction)




toy_text = [line.strip().split(',', 1)[1] for line in open(
        toy, "r", encoding='utf-8-sig')][1::]
toy_list = [text_to_word_sequence(  s, filters='', lower=True, split=" ") for s in toy_text]
toy_train = np.array( [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in toy_list])
prediction = model.predict(toy_train)

toy_train_BoW = tokenizer.texts_to_matrix(toy_text,mode='count')
BOW_model.predict(toy_train_BoW)

model.save_weights(BOW_model,'BOW_model.h5')
BOW_model = load_model('C:/Users/user/Desktop/BOW_model.h5')
