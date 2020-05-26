import pandas as pd
import tweepy
import csv
import nltk
import re
import string
from nltk.tokenize import TweetTokenizer
from contractions_eng import contractions_dict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn import metrics
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from imblearn.over_sampling import RandomOverSampler
from tensorflow.contrib.rnn import *
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import LabelEncoder
import h5py
from keras.models import load_model


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

def get_embed_mat(embedding_path):
    embed_size = 300
    max_features = 20000
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path,encoding="utf8"))
    tokenizer = Tokenizer(num_words=20000,lower = True, filters = '')
    word_index = tokenizer.word_index
    nb_words = max_features
    print(nb_words)
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix

def build_model1(X_train,y_one_hot,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    embed_size = 300
    max_features = 20000
    max_len = 50
    
    file_path = "model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)
    embedding_path = "crawl-300d-2M.vec"
    embedding_matrix = get_embed_mat(embedding_path)
    print(embedding_matrix.shape)
    
    inp = Input(shape = (max_len,))
    x = Embedding(20001, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm1 = GlobalAveragePooling1D()(x1)
    max_pool1_lstm1 = GlobalMaxPooling1D()(x1)
    
    
    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    
    
    x = concatenate([avg_pool1_lstm1, max_pool1_lstm1,
                    avg_pool1_lstm, max_pool1_lstm])
    #x = BatchNormalization()(x)
    x = Dropout(0.1)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.1)(Dense(64,activation='relu') (x))
    x = Dense(2, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_one_hot, batch_size = 128, epochs = 20, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model


    
#Main
def main():
    
    # Importing the dataset
    df_train = pd.read_csv("C:\python\\training.1600000.processed.noemoticon.csv ", encoding = 'latin1')
    df_train = df_train.drop(df_train[df_train.num_of_words < 1].index)
    df_test = pd.read_csv("C:\python\\testdata.manual.2009.06.14.csv ", encoding = 'latin1')
    # adding the test and train data tweets
    full_text = list(df_train['TWEET'].values) + list(df_test['TWEET'].values)
    #tokenizerr
    tokenizer = Tokenizer(num_words=20000,lower = True, filters = '')
    tokenizer.fit_on_texts(full_text)
    #train tokenizer
    train_tokenized = tokenizer.texts_to_sequences(df_train['TWEET'])
    #test tokenizer
    test_tokenized = tokenizer.texts_to_sequences(df_test['TWEET'])
    max_len = 50
    #pad sequences
    X_train = pad_sequences(train_tokenized, maxlen = max_len)
    X_test = pad_sequences(test_tokenized, maxlen = max_len)
    labels_train = df_train.LABEL
    #label encoder
    lb=LabelEncoder()
    labels1_train=lb.fit_transform(labels_train)
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_one_hot_train = one_hot_encoder.fit_transform(labels1_train.reshape(-1,1))
    #print(y_one_hot_train.Shape)
    model = build_model1(X_train,y_one_hot_train,lr = 1e-3, lr_d = 0, units = 128, dr = 0.5)
    # model predict values
    pred1 = model.predict(X_test, batch_size = 1024, verbose = 1)
    # model probabilities
##    pred1_prob = model.predict_proba(X_test, batch_size = 1024, verbose = 1)
##    # prediction by probabilities
##    emotions_predicted = []
##    for i in range(pred1_prob.shape[0]):
##        if (pred1_prob[i][0] <= 0.3 and pred1_prob[i][1] <= 0.3):
##            emotions_predicted.append(int(2))
##        else:
##            emotions_predicted.append(np.argmax(probabilities[i,:]))
##    for i in  range(len(emotions_predicted)):
##        if emotions_predicted[i] == 1:
##            emotions_predicted[i] = 4
##    
    predictions1 = np.round(np.argmax(pred1, axis=1)).astype(int)
    y_test= df_test['LABEL']
    y_test1=lb.fit_transform(y_test)
    print("The accuracy score",metrics.accuracy_score(y_test1,predictions1))
    print("The classification report",metrics.classification_report(y_test1,predictions1))
    df_test['pred_rnn'] = y_test1
    #df_test['pred_prob_rnn'] = emotions_predicted
    df_test.to_csv('testdata.manual.2009.06.14.csv')
    
    


if __name__ == "__main__":
    main()
    
