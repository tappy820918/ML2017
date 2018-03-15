import sys
import csv
import numpy as np
import pandas as pd
import keras
from keras import regularizers,models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Merge, Input, Dot, Reshape, merge, Dropout, Add
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam

from keras.layers import *
from numpy import bincount, ravel, log
from keras.models import Model, Sequential
from sklearn.utils import shuffle
'''
train_path = 'C:/Users/user/Desktop/train.csv'
test_path = 'C:/Users/user/Desktop/test.csv'
movies_path = 'C:/Users/user/Desktop/movies.csv'
users_path = 'C:/Users/user/Desktop/users.csv'
filepath = 'C:/Users/user/Desktop/weight/weight'
predict_path = 'C:/Users/user/Desktop/PREDICT.csv'


train_path = 'train.csv'
test_path = 'test.csv'
movies_path = 'movies.csv'
users_path = 'users.csv'

'''
test_path    = sys.argv[1]
predict_path = sys.argv[2]
movies_path  = sys.argv[3]
users_path   = sys.argv[4]



train  = shuffle(pd.read_csv(train_path))
test   = pd.read_csv(test_path)

VALIDATION_SPLIT = 0.15

K_FACTORS = 200
max_userid = train['UserID'].drop_duplicates().max()
max_movieid = train['MovieID'].drop_duplicates().max()

train['Rating'] = train['Rating'] / 5


def MF_model(n_users, n_items, latent_dim=K_FACTORS):
    user_input = Input(shape=[1], name='UserID')
    item_input = Input(shape=[1], name='MovieID')
    user_vec = Embedding(n_users, latent_dim)(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items + 1, latent_dim)(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1)(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items + 1, 1)(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='Adam')
    return (model)

def write_result(prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["TestDataID", "Rating"])
    for i in range(len(prediction)):
        s.writerow([i + 1, abs(prediction[i])])
    text.close()

ACCearlyStopping = EarlyStopping(
        monitor='val_loss', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


MFModel = MF_model(max_userid, max_movieid)
train_history = MFModel.fit([train['UserID'], train['MovieID']], train['Rating'], verbose=1, batch_size=10000, epochs=10,
                            callbacks=[checkpoint, ACCearlyStopping], validation_split=0.1, class_weight='auto')

ANS = MFModel.predict([test['UserID'], test['MovieID']])
ANS2 = np.transpose(ANS)[0]
write_result(ANS2*5)

'''
models.save_model(train_history.model,'hw5_MFmodel.h5')


MODEL = load_model('hw5_MFmodel.h5')
ANS = MODEL.predict([test['UserID'], test['MovieID']])
write_result(np.transpose(ANS)[0])
'''