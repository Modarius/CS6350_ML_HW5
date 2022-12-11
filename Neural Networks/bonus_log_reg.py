# This code is heavilly based on code written for CS6440 Image Processing homework
# which was based on the code provided in pytorch-tutorial-main
# which was provided as an example for the homework

from collections import OrderedDict
from time import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train(x, y, w, v, gamma):
    for i in range(len(y)):
        # get a single datapoint
        curr_y = y[i] 
        curr_x = np.append([1], x[i]) # augment the datapoint
        grad = len(y) * (-curr_y*curr_x/(1 + np.exp(curr_y * np.dot(w,curr_x))) + 2/v*w) # calculate the gradient
        w = w - gamma * grad # update the weights based on the gradient
    return w

def test(x, y, w, v, gamma):
    x_aug = np.vstack((np.ones(len(y)),x.T)).T # augment the data with 1 for the bias
    pred = np.sign(np.matmul(w, x_aug.T)) # prediction is the sign of w.T * x
    correct = np.sum(y == pred) # count the correct predictions
    error = 1-correct/len(y) # get the error
    return error
            
def importData(filename, column_labels=None, index_col=None, header=None):
    # index_col is a number indicating which column, if any, is the index into the data
    # header is the line of the data if any that where column labels are indicated
    terms = pd.read_csv(filename, sep=',', names=column_labels, index_col=index_col, header=header, dtype=np.float32) # read in the csv file into a DataFrame object , index_col=index_col
    # if needed any processing can be done here
    terms['label'].where(terms['label'] != 0, -1, inplace=True) # change labels to be {-1, 1}
    return terms

def main():
    p_var = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100] # list of variances to test
    gamma_0 = 0.03 # from homework 4
    d = 0.0001 # from homework 4
    gamma = 1E-8 # gamma_0/(1+gamma_0/d) # gamma calulation
    EPOCHS = 100

    for v in p_var: # for each variance
        print("Variance: " + str(v), end='', flush=True)
        tic = time() # keep time and see how long it takes

        # code to import data
        dir_path = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/a/5137509
        train_data = importData(dir_path + '/bank-note/train.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        train_Y = train_data.pop('label').to_numpy()
        train_X = train_data.to_numpy()

        test_data = importData(dir_path + '/bank-note/test.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        test_Y = test_data.pop('label').to_numpy()
        test_X = test_data.to_numpy()

        w = np.append([1], np.zeros(4)) # make an empty weight vector with a 1 for the bias (w_0 = 1, x_0 = 1)
        rand = np.random.default_rng()
        indices = np.arange(train_data.shape[0])
        train_errors = np.zeros(EPOCHS) # empty array to store errors
        test_errors = np.zeros(EPOCHS) # empty array to store errors
        for i in range(EPOCHS):
            rand.shuffle(indices) # get a new ordering of data
            train_X = train_X[indices] # randomize the data but the same for x and y so they match up
            train_Y = train_Y[indices]

            w = train(x=train_X, y=train_Y, w=w, v=v, gamma=gamma) # run data through the training algorithm to get a good weight vector
            train_errors[i ]= test(x=train_X, y=train_Y, w=w, v=v, gamma=gamma) # test the weight vector
            test_errors[i] = test(x=test_X, y=test_Y, w=w, v=v, gamma=gamma) # test the weight vector

        print(' done, time: ' + str(time() - tic), flush=True)
        np.savetxt('error_train_v' + str(v) + '.csv', train_errors, fmt='%.4f', delimiter=',')
        np.savetxt('error_test_v' + str(v) + '.csv', test_errors, fmt='%.4f', delimiter=',')
    return

if __name__ == "__main__":
    main()