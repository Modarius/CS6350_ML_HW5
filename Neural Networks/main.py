import os
import pandas as pd
import numpy as np

class neuralNet:
    def __init__(self, in_width=3, hl_width=3, out_width=1, initialize='rand'):
        self.input_width = in_width
        self.hidden_width = hl_width
        self.output_width = out_width

        if (initialize=='zeros'): # hl_width - 1 because of constant bias term
            self.weight1 = np.zeros((hl_width-1, in_width)) # rows = num nodes going to, cols = num nodes coming from
            self.weight2 = np.zeros((hl_width-1, hl_width)) # rows = num nodes going to, cols = num nodes coming from
            self.weight3 = np.zeros((out_width, hl_width))  # rows = num nodes going to, cols = num nodes coming from
        elif (initialize=='rand'):
            self.weight1 = np.random.rand(hl_width-1, in_width)
            self.weight2 = np.random.rand(hl_width-1, hl_width)
            self.weight3 = np.random.rand(out_width, hl_width)
        pass

    def predict(self, x):
        z1 = sigmoid(np.array(1, sum(self.weight1[:,0] * x ), sum(self.weight1[:,1] * x)))
        z2 = sigmoid(np.array(1, sum(self.weight2[:,0] * z1), sum(self.weight2[:,1] * z1)))
        return sum(self.weight3 * z2)

def squareLoss(pred, truth):
    return 1/2 * np.square(pred-truth)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def dSigmoid(z):
    s = sigmoid(z)
    return (1-s) * s

def importData(filename, column_labels=None, index_col=None, header=None):
    # index_col is a number indicating which column, if any, is the index into the data
    # header is the line of the data if any that where column labels are indicated
    terms = pd.read_csv(filename, sep=',', names=column_labels, index_col=index_col, header=header) # read in the csv file into a DataFrame object , index_col=index_col
    # if needed any processing can be done here
    terms['label'].where(terms['label'] != 0, -1, inplace=True) # change labels to be {-1, 1}
    return terms

def main(runs):
    EPOCHS = 10
    errors = np.zeros(EPOCHS)
    for test_run in range(runs):
        dir_path = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/a/5137509
        train = importData(dir_path + '/bank-note/train.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        train_Y = train.pop('label').to_numpy()
        train_X = train.to_numpy()

        test = importData(dir_path + '/bank-note/test.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        test_Y = test.pop('label').to_numpy()
        test_X = test.to_numpy()

        error = np.zeros(EPOCHS)
        rand = np.random.default_rng()
        indices = np.arange(train.shape[0])
        for i in range(0,EPOCHS):
            rand.shuffle(indices) # get a new ordering of data
            train_X = train_X[indices] # randomize the data but the same for x and y so they match up
            train_Y = train_Y[indices]
            error[i] = 'PLACEHOLDER'
        errors[test_run] = error
    errors /= runs
    return

if __name__ == "__main__":
    main()