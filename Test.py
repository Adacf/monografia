# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 07:44:10 2017

@author: Ada
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt
import tkinter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
import math


#################################################################


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


# Simple normalization
def simpleNormalization(data):
    numcols = len(data[0])

    for x in range(0, numcols):  # colunas
        col = data[:, x]
        lvmax = max(col)
        lvmin = min(col)
        lvrange = (lvmax - lvmin) if (lvmax - lvmin) != 0 else 1
        for y in range(0, len(col)):  # linhas
            data[y, x] = (data[y, x] - lvmin) / lvrange

        if( x == numcols-1):
            print("lvmin:" + str(lvmin) +  " lvrange:" + str(lvrange))
            #print(data[y, x])

    #print(data)

def plot(test_data, series):
    X = np.arange(len(test_data))


    plt.plot(X, test_data, 'bs')
    plt.show()

    for s in series:
        plt.plot(X, s, 'bs')
    plt.show()


def load(test):

    with open('data.csv') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append([float(x) for x in row if is_number(x)])

        testando = data.pop()

        print(testando.pop())
        test.append(testando)

        data = np.array(data[1:])
        simpleNormalization(data)
        return data


 #def runMlp(train_data, train_labels, test_data, test_labels):
  #  mlp = MLPClassifier(solver='sgd', learning_rate='adaptive', learning_rate_init=0.02, max_iter=10000,
   #                     activation='tanh', momentum=0.9, shuffle=True)
   # mlp.fit(train_data, train_labels)
   # prediction = mlp.predict(test_data)
   # return prediction


def runSVM(train_data, train_labels, test_data, test_labels):

    svr_lin = svm.SVR(kernel="linear", C=1e3)
    svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = svm.SVR(kernel='rbf',C=1e3, gamma=0.1)

    clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
                  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    #clf.fit(train_data, train_labels).values

    svr_rbf.fit(train_data,train_labels)
    svr_poly.fit(train_data,train_labels)
    svr_lin.fit(train_data,train_labels)
    prediction = svr_poly.predict(test_data)

    print(test_data)
    print("Result" + str(svr_rbf.predict(test_data)))
    return prediction


def runLSTM(trainX, trainY, testX, testY, scaler):

    print(np.shape(trainY))
    model = Sequential()
    model.add(LSTM(400,input_shape=(2,3)))
    model.add(Dense(2))
    model.compile(loss='mae',optimizer='adam')
    history = model.fit(trainX, trainY, epochs=50, batch_size=1, validation_data=(testX, testY), verbose=2, shuffle=False)

    #plt.plot(history.history['loss'], label='train')
   # plt.plot(history.history['val_loss'], label='test')
   # plt.legend()
   # plt.show()

    yhat = model.predict(testX)


    plt.plot(yhat,label='results')
    plt.plot(testY, label="realResults")
    plt.legend()
    plt.show()

    test_X = testX.reshape((testX.shape[0], testX.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = testY.reshape((len(testY), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = math.sqrt(metrics.mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

#read dataset
def readDataSet():

    with open('oneinstance1.csv') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append([float(x) for x in row if is_number(x)])

    return data

def createplotspecific(data):

    dataset = data[1]

    npdataset = np.array(dataset[0:-1])

    label = dataset[len(dataset)-1]

    #separate by variables

    type = npdataset[0]  # type - GDU or GDT
    print(len(npdataset))
    rightwheel = np.log(npdataset[0:400])

    leftwheel = np.log(npdataset[400:800])

    rightbearing = np.log(npdataset[800:1200])

    leftbearing = np.log(npdataset[1200:1600])

    speed = npdataset[1600:2000]

    weight = npdataset[2000:2400]

    #plot each variable

    plt.figure()
    plt.subplot(6,1,1)
    plt.plot(rightwheel)
    plt.title("rightwheel", y=0.20, loc='right')

    plt.subplot(6, 1, 2)
    plt.plot(leftwheel)
    plt.title("leftwheel", y=0.20, loc='right')

    plt.subplot(6, 1, 3)
    plt.plot(rightbearing)
    plt.title("rightbearing", y=0.20, loc='right')

    plt.subplot(6, 1, 4)
    plt.plot(leftbearing)
    plt.title("leftbearing", y=0.20, loc='right')

    plt.subplot(6, 1, 5)
    plt.plot(speed)
    plt.title("speed", y=0.20, loc='right')

    plt.subplot(6, 1, 6)
    plt.plot(weight)
    plt.title("weight", y=0.10, loc='right')

    plt.show()
    newdata = []

    rightbearing1 = []
    speed1 = []
    weight1 = []

    for i in range(0,3):
        values = data[i]
        rightbearing1 = values[800:1200]
        speed1 = values[1600:2000]
        weight1 = values[2000:2400]

        for j in range(0,400):
            newdata.append([speed1[j],weight1[j],np.log(rightbearing1[j])])

    return np.array(newdata)

def series_to_supervised_copy(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pandas.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = pandas.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def main():

    dataset = readDataSet()

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    ## TESTE LSTM ##

        ## Create one instance
    oneinstance = createplotspecific(dataset)

    values = oneinstance.astype("float32")

        ## Normalize
    scaled = scaler.fit_transform(values)

        ## Change to a specific dataset
    reframed = series_to_supervised_copy(scaled,1,1)

        ## Drop columns we don't want to predict
    reframed.drop(reframed.columns[[3,4]],axis=1, inplace=True)

        ## Get just the values
    values = reframed.values

        ## Separate train and test
    train = values[0:800]
    test = values[799:1200]
    train_X = train[:, :-1]
    train_Y = train[:, -1]
    test_X = test[:, :-1]
    test_Y = test[:, -1]

    train_X = train_X.reshape((400,2,train_X.shape[1]))
    train_Y = train_Y.reshape((400,2))

    test_X = test_X.reshape((200,2,3))
    test_Y = test_Y.reshape((200,2))

    runLSTM(train_X,train_Y,test_X,test_Y,scaler)

#    runSVM(train_X,train_Y,test_X,test_Y)
    # FIM TESTE

    #labels = scaled[:,-1]
    #scaled = scaled[:,0:-1]

    #series_to_supervised(scaled,1,1)
    # print(labels)

   # values = scaled.values


    #train_data, test_data, train_labels, test_labels = train_test_split(scaled, labels, train_size=0.5)

    #runLSTM(train_data,train_labels,test_data,train_labels)

    #createplotspecific(scaled[301])



    #dataset = load(data)
    #data = dataset[:, 0: -1]
    #labels = dataset[:, -1]  # last colum
    #train_data, test_data, train_labels, test_labels = train_test_split(data, labels,train_size=0.5)

    #prediction = runSVM(train_data, train_labels, test_data, test_labels,data)

    #for val in range(len(test_labels)):
     #   print(str(test_labels[val]) + ',' + str(prediction[val]))

    #plot(test_labels, [prediction])

main()