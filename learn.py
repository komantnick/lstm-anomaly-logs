import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
import sys
import os
import json
import numpy
import optparse
from keras.callbacks import TensorBoard
from keras.layers import SpatialDropout1D,Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from urllib.parse import urlparse
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import np_utils
#from sklearn.preprocessing import MinMaxScaler


#обработка данных
np.random.seed(1234)


def get_url(url):
    return urlparse(url).hostname

def delete_slash(text):
    return text.replace(' ', '')

#script about logs anomaly

df = pd.read_csv("r6.2/http.csv",nrows=100000,usecols=["user", "url", "activity"],)
df['url'] = df['url'].apply(get_url)
df['activity'] = df['activity'].apply(delete_slash)
df['load-data'] = df['user']+" "+ df['url']
del(df['url'])
del(df['user'])
#df['list']=df['user']+" "+df['url']
ds = df.sample(frac=1).values
# ds = ds.astype('float32')
# #normalizing dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds = scaler.fit_transform(ds)


#converting to array
X=ds[:,1]
Y=ds[:,0]
#exit()
#tokenize our array
t = Tokenizer()
t2 = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(X)
# summarize what was learned

#print(t.word_docs)
encoded = t.texts_to_sequences(X)
num_words = len(t.word_index)+1

#trunkate data
#print(len(encoded))

max_len=10

for sublist in encoded:
     for index, value in enumerate(sublist):
         sublist[index]=value/num_words
         #print(sublist[index])
encoded  = sequence.pad_sequences(encoded , maxlen=max_len, dtype='double')
t2.fit_on_texts(Y)
encoded2 = t2.texts_to_sequences(Y)
encoded2 = [item for sublist in encoded2 for item in sublist]
# print(encoded2)
# exit()
train_size = int(len(ds) * .75)
X_train, X_test = encoded[0:train_size], encoded[train_size:len(encoded)]
Y_train, Y_test = encoded2[0:train_size], encoded2[train_size:len(encoded2)]
# for a in encoded:
#     print(a)
#не цифры от 0 до 1
nb_classes=4
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
# for a in Y_test:
#     print(a)
# exit()

#making an unsupervised neural network model
def lstm_model(array,test_array,num_words,num_of_cell):
    embedding_vecor_length = 10
    nb_classes=4
    model = Sequential()
    model.add(Embedding(num_words, embedding_vecor_length, input_length=10))
    model.add(LSTM(num_of_cell))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,Y_train,nb_epoch=1,batch_size=128)
    predicted = model.predict(X_test)
    # print(predicted.size)
    # exit()
    predictedk = model.predict( X_test)
    predicted = np.reshape(predicted, (predicted.size//nb_classes,nb_classes))
    predictedk = np.reshape(predictedk, (predictedk.size//nb_classes,nb_classes))
    scores = model.evaluate(X_test,Y_test, batch_size=32)
    print("Accuracy of LSTM: %.2f%%" % (scores[1]*100))


    #making a plot

    plt.figure(1)
    plt.subplot(311)
    plt.title("Actual-B,Predicted-G Test Signal")
    plt.plot(Y_test[:len(Y_test)], 'b')
    plt.subplot(312)
    plt.title("Predicted Signal")
    plt.plot(predicted[:len(Y_test)], 'g')
    plt.subplot(313)
    plt.title("Squared Error")
    mse = ((Y_test - predicted) ** 2)
    plt.plot(mse, 'r')
    plt.show() 

    for a in predictedk:
        print(a)  


    # model = Sequential()
    # model.add(LSTM(5, input_dim=10000, return_sequences=True))
    # model.add(LSTM(5, return_sequences=True))
    # model.add(Flatten())
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(array, array)

    #мин-1, макс-200. максимальное значение. в каждой переменной разделить на максимальное значение на переменных доли!
    #неправильно преобразованные аднные! 200 значений userов-каждый user:массив из 200 элементов:1 или 0

    #нужно по долям!

def simplernn_model(array,test_array,num_words,num_of_cell):
    embedding_vecor_length = 10
    model = Sequential()
    model.add(Embedding(num_words, embedding_vecor_length, input_length=10))
    model.add(SimpleRNN(num_of_cell))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,Y_train, batch_size=128)
    scores = model.evaluate(X_test,Y_test, batch_size=32)
    print("Accuracy of SImple RNN: %.2f%%" % (scores[1]*100))

#графики! которые показывают. в конце встроить функцию поиска аномалии. 
# на каждое новое действие сравнивает с тем, что предсказывает!
#создать лог пользователей 10 download'ов подряд - это не нормально
#preditcion! пользователь делал upload upload далее модель. система должна 
#предсказывает следующий шаг и с тем что реально. если не похоже на то что на реальное- то. 90-visit,70-upload,20-upload.

#подача на вход цепочки событий. система берет цепочку и смотрит-какие события в цпочке непраивльные
#1 событие-по 1 второе-по 1 и 2 третье->где-то предсказания будут расходиться. систсема должна помечать как аномальные события.
#нужно трешхолд-разница в векторах предсказания.
#говорит вероятность-upload нижняя вероятность-это плохо
#трешхолд 3 значения. если равно самому непредсказуемо-значит аномалия
#несколько сценариев-например 5
#они должны находиться. что находят все эти сценарии!

#документация. как ищется аномалия?почему не средее скользящее? рамка считывания, скользящее среднее

#модель attention

def grumodel(array,test_array,num_words,num_of_cell):
    embedding_vecor_length = 10
    model = Sequential()
    model.add(Embedding(num_words, embedding_vecor_length, input_length=10))
    model.add(GRU(num_of_cell))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,Y_train,epochs=1, batch_size=128)
    scores = model.evaluate(X_test,Y_test, batch_size=32)
    print("Accuracy of LSTM: %.2f%%" % (scores[1]*100))

train_size = int(len(encoded) * .75)
train, test = encoded[0:train_size], encoded[train_size:len(encoded)]
print("Test-1:LSTM 100 cells on 50000 elements")
lstm_model(train,test,num_words,100)
# print("Test-2:GRU 100 cells on 50000 elements")
# grumodel(train,test,num_words,100)
# print("Test-3:SimpleRNN 100 cells on 50000 elements")
# simplernn_model(train,test,num_words,100)
# print("Test-4:LSTM 10 cells on 50000 elements")
# lstm_model(train,test,num_words,10)
# print("Test-5:GRU 10 cells on 50000 elements")
# grumodel(train,test,num_words,10)
# print("Test-6:SimpleRNN 10 cells on 50000 elements")
# simplernn_model(train,test,num_words,10)

#def plotgraph():