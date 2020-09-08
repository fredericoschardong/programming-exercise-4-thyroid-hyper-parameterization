import sys
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.metrics import classification_report

#supress warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

#load data
dataset = loadmat('thyroid.mat')

#prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['Y'], stratify=dataset['Y'], random_state = 1, test_size = 0.3)

#histogram
for i in range(5):
    x = X_train[:,i]
    hist, bins = np.histogram(x, bins='auto')
    plt.plot(bins[:hist.size], hist / np.sum(hist))
    #print(i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

plt.xlabel('Values')
plt.ylabel('Proportions')
plt.savefig('Histogram before normalization.png')
plt.clf()

for scaler in [MinMaxScaler(), PowerTransformer()]:
    #normalization
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #new histogram with normalized data
    for i in range(5):
        x = X_train[:,i]
        hist, bins = np.histogram(x, bins='auto')
        plt.plot(bins[:hist.size], hist / np.sum(hist))
        #print(scaler, i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

    plt.xlabel('Values')
    plt.ylabel('Proportions')
    plt.savefig('Histogram after normalization with %s.png' % scaler)
    plt.clf()

    with open('result_tables with %s.tex' % scaler, 'w') as f:
        for max_iter in [100, 1000, 10000]:
            for solver in ['lbfgs', 'sgd', 'adam']:
                #parameter search space
                parameters = {'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (5,5), (10,10), (20,20), (5,10,20), (20,10,5)],
                              'activation': ['identity', 'logistic', 'tanh', 'relu'], 
                              'alpha': [0.00001, 0.0001, 0.001, 0.01], 
                              'random_state': [1],
                              'max_iter': [max_iter],
                              'solver': [solver]}

                #use f1 to rank parameters, all cores and 5-cross fold validation
                clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
                clf.fit(X_train, y_train.ravel())
                y_true, y_pred = y_test, clf.predict(X_test)
                
                #remove unewanted data for latex table
                report = classification_report(y_true, y_pred, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df = df.iloc[:, :-1]
                df.drop(df.index[[3,4]], inplace=True)

                #inject the configuration as a comment in the caption field to help me later
                print(df.to_latex(float_format="%.2f", decimal=',', caption='}%% %s' % clf.best_params_), file=f)
                
                f.flush()
