#!/usr/bin/env python3.6

import os
import sys
sys.path.append('../')
from LRC import LRC 
from sklearn.decomposition import PCA
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
import csv
from knn import KNN
import sys

# Load data
data = []
FILENAME = os.path.dirname(__file__) + '/IRIS.csv'
f = open(FILENAME)
reader = csv.reader(f)
for row in reader:
    data.append(row)

# Test
series = 1000
goodOnesLRC = 0.0
goodOnesKNN = 0.0
allOnes = 0.0

for i in range(series):

    sys.stdout.write("\r\x1b[K" + "Prediction progress: " 
                            + str(int((i+1)/series*100)) + "%")
    sys.stdout.flush()

    np.random.shuffle(data)
    inputs = np.array(np.delete(data, 8, 1), dtype='f')
    targets = np.array(data)[:,8]
    learn_data = inputs[:80]
    predict_data = inputs[80:]

    predictorLRC = LRC(
        learn_data = learn_data, 
        targets = targets[:80]
    )

    predictorKNN = KNN(
		k = 7, 
		algoritm = 'ball_tree', 
		learn_data = learn_data,
		targets = targets
	)
    
    predictionsLRC = predictorLRC.predict(predict_data)

    predictionsKNN = predictorKNN.predict(predict_data)

    for i in range(len(predictionsLRC)):
        allOnes += 1

        if predictionsLRC[i] == targets[80+i]:
            goodOnesLRC += 1

        if predictionsKNN[i] == targets[80+i]:
            goodOnesKNN += 1

print("\n")
print("LRC test error: " + str(1-goodOnesLRC/allOnes))
print("KNN test error: " + str(1-goodOnesKNN/allOnes))
