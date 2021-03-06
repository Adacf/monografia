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


def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


# fit an LSTM network to training data
def fit_lstm(trainX, trainY, n_lag, n_seq, n_batch, nb_epoch, n_neurons):

    # design network
    model = Sequential()

    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, trainX.shape[1], trainX.shape[2]), stateful=True))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # fit network

    #for i in range(nb_epoch):
     #   history = model.fit(trainX, trainY, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
      #  model.reset_states()

    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2, shuffle=False)
    return model


def make_forecasts(model, n_batch, train_X,trai_Y, test_X, test_Y, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test_X)):
        X = test_X[i,0:3]
        Y = test_Y[i, 3:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def inverse_difference(last_ob, forecast):
	# invert first forecast
	enverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pandas.Series(diff)

def runLSTM(trainx, trainy, testx, testy, scaler):

    print(np.shape(trainx))

    test = trainx.reshape(3,699,3)
    print(test)
    print(np.shape(trainy))
    testa = trainy.reshape(3,699)
    print(testa)
    n_neurons = 3;

    model = Sequential()
    model.add(LSTM(n_neurons,input_shape=(3,3)))
    model.add(Dense(3))

    model.compile(loss='mae',optimizer='adam',metrics=['accuracy'])
    history = model.fit(trainx, trainy, epochs=200, batch_size=10, verbose=2, shuffle=False)

    #plt.plot(history.history['loss'], label='train')
   # plt.plot(history.history['val_loss'], label='test')
   # plt.legend()
   # plt.show()
  #  testx = testx.reshape(300,1,3)
  #  print(np.shape(testx))
    yhat = model.predict(testx)
    yhat = yhat.reshape(300,1)
    yhat = yhat[:100]
    testy = testy.reshape(300,1)
    testy = testy[:100]

    plt.plot(yhat,label='results')
    plt.plot(testy, label="realResults")
    plt.legend()
    plt.show()

    print(np.shape(testx))
    testx = testx.reshape(300,3)

    testx = testx[:100]



def readDataSet(size_samples):

    var_type = "tipo"
    var_wr = "wr"
    var_wl = "wl"
    var_br = "br"
    var_bl = "bl"
    var_speed = "s"
    var_weight = "p"

    type = []
    wheelright = []
    wheelleft = []
    bearingright = []
    bearingleft = []
    speed = []
    weight = []

    series = pandas.read_csv('instances.csv', header=0,index_col=0, squeeze=True)

    #get headers
    headers = series.columns

    #get values
    values = series.values

    data = []

    for row in values:
        for index, column in enumerate(row):
            type.append(row[index])
            if "wr" in headers[index]:
                wheelright.append(column)
            if "wl" in headers[index]:
                wheelleft.append(column)
            if "br" in headers[index]:
                bearingright.append(column)
            if "bl" in headers[index]:
                bearingleft.append(column)
            if "s" in headers[index]:
                speed.append(column)
            if "p" in headers[index]:
                weight.append(column)
        # transform data to be stationary
        diff_series = difference(bearingright, 1)
        diff_values = np.array(diff_series.values)
        for index, values in enumerate(diff_values):
            data.append([speed[index], weight[index], bearingright[index]])

    data = np.array(data)

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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

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


def create_tuples(data, n_samples):

    # Separate data to predict
    newdata = []

    for i in range(0,n_samples):
        values = data[i]
        rightbearing1 = values[800:1200]
        speed1 = values[1600:2000]
        weight1 = values[2000:2400]

        for j in range(0,400):
            newdata.append([speed1[j],weight1[j],rightbearing1[j]])

    return np.array(newdata)

def prepare_data(series, n_test, n_lag, n_seq, n_samples):

    values = series.astype("float32")

    # normalize
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print(scaled)
    sizei = 0
    sizef = int(len(scaled)/n_samples)
    value = sizef
    newdataset = [];

    # Change to a specific dataset
    for i in range(0, 3):
        supervised = series_to_supervised(scaled[sizei:sizef], 1, 10)
        sizei = sizef
        sizef += value
        # Drop columns we don't want to predict
        #6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31
        supervised.drop(supervised.columns[[3, 4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]], axis=1, inplace=True)

        values = supervised.values
        for tuple in values:
            newdataset.append(tuple)

    newdataset = np.array(newdataset)

    # split between train and test
    train, test = newdataset[0:-n_test], newdataset[-n_test:]

    #train_X, train_Y, test_X, test_Y = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

   # print(np.shape(train_X))
    # split between samples
    #train_Y = train_Y.reshape((int(len(train_X)/n_samples), n_samples))
    #train_X = train_X.reshape((int(len(train_X)/n_samples), n_samples, train_X.shape[1]))

    #test_Y = test_Y.reshape((int(len(test_X)/n_samples), n_samples))
    #test_X = test_X.reshape((int(len(test_X)/n_samples),n_samples,3))
    #test_X = test_X.reshape((400, n_samples, 3))
    return train,test


def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]


def make_forecasts(train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def main():

    # configure
    n_lag = 1
    n_seq = 3
    n_test = 300
    n_samples = 3

    # read dataset #
    dataset = readDataSet(n_samples)

    # TESTE LSTM #

    # prepare data #
    train,test = prepare_data(dataset, n_test, 1, 3, 3)

    trainX = train[:,0:3]
    trainY = train[:,3:]

    trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
    testX = test[:,0:3]
    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
    testY = test[:,3:]

    testando = []
    testando.append(testX[0])
    testando.append(testX[1])
    testando.append(testX[2])

    testandoy = []
    testandoy.append(testY[0])
    testandoy.append(testY[1])
    testandoy.append(testY[2])
    testando = np.array(testando)
    print(np.shape(testando))
    #print(testY)
    #print(np.shape(testY))
    # run LSTM #
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #runLSTM(train_X,train_Y,test_X,test_Y,scaler)
    model = fit_lstm(trainX,trainY,1,10,1,50,1)
    #forecasts = make_forecasts(model,1, train_X, train_Y, test_X,test_Y, 1, 10)

    yhat = model.predict(testando,1)
    #yhat = yhat.reshape(300, 1)
    #yhat = yhat[:100]
    #testy = testY.reshape(300, 1)
    #testy = testy[:100]
    #print(list(yhat))
    #print(list(testY))


    plt.plot(yhat, label='results')
    plt.plot(testandoy, label="realResults")
    plt.legend()
    plt.show()
    # END TESTE #

main()