import math
import scipy.stats as stats
import numpy as np
import sys

class LRC:
    ''' LRC stands for linear ranking classificator '''

    def __init__(self, learn_data, targets, log=False):
        ''' learn_data - learning data as np.array, columns - attributes, rows - records.
        targets - categorical type
        log - set as 'True' if you want to see logs'''
        self.learn_data = learn_data
        self.learn_data_T = np.transpose(self.learn_data)
        self.targets = targets
        self.log = log

    def predict(self, data):

        if len(self.learn_data[0]) != len(data[0]):
            print('Learn data and predict data do not have the same number of attributes.')
            return

        return_predictions = [0 for x in range(len(data))]
        classes = np.unique(self.targets)
        numOfClasses = len(classes)
        classesDist = [[[0.0 for x in range(numOfClasses)] for y in range(numOfClasses)] for y in range(len(data))]
        classesData = []

        for i in range(len(classes)):
            classesData.append(self.learn_data[self.targets == classes[i]])

        for x in range(numOfClasses):
            for y in range(numOfClasses):
                
                if x < y+1:
                    continue

                targets = []
                learn_data = []
                targets.extend([1 for z in range((self.targets == classes[x]).sum())])
                targets.extend([0 for z in range((self.targets == classes[y]).sum())])
                a1 = np.array(self.learn_data[self.targets == classes[x]])
                a2 = np.array(self.learn_data[self.targets == classes[y]])
                learn_data = np.concatenate((a1,a2))
                targets = np.array(targets)
                bin_threshold = (targets == 0.0).sum()/float(len(targets))
                learn_data_T = np.transpose(learn_data)

                for i in range(len(data)):
                    series_predictions = [[0 for l in range(2)] for b in range(len(learn_data_T))]

                    for j in range(len(learn_data_T)):
                        
                        series_predictions[j] = self.predictClass(
                            targets,
                            learn_data_T[j],
                            data[i][j]
                        )

                    distances = self.calcDistances(series_predictions, bin_threshold)
                    classesDist[i][y][x] = distances[0]
                    classesDist[i][x][y] = distances[1]

                    if self.log:
                        sys.stdout.write("\r\x1b[K" + "Prediction progress: " 
                            + str(int((i+1)*100/len(data))) + "%")
                        sys.stdout.flush()

        for k in range(len(data)):
            distances = [0 for x in range(numOfClasses)]

            for i in range(numOfClasses):
                distances[i] = sum(classesDist[k][i])

            return_predictions[k] = classes[distances.index(min(distances))]

        if self.log:
            print("\n")

        return return_predictions

    def calcDistances(self, predictions, bin_threshold):
        rSum = 0.0
        dist1 = 0.0
        dist2 = 0.0

        for i in range(len(predictions)):
            r = abs(predictions[i][1])
            rSum += abs(r)*abs(r)
            dist1 += predictions[i][0]/bin_threshold * r
            dist2 += (1-predictions[i][0])/(1-bin_threshold) * r
        
        dist1 = dist1/pow(rSum, 0.5)
        dist2 = dist2/pow(rSum, 0.5)

        return [dist1, dist2]

    def predictClass(self, x, y, yN):
        pearsonr = stats.pearsonr(x, y)
        r = pearsonr[0]
        p = pearsonr[1]

        y = np.append(y, yN)
        y = stats.rankdata(y)-1
        yN = int(y[-1])
        predictX = yN/(len(y)-1)
        
        if r<0:
            predictX = 1.0 - predictX

        return [predictX, r, p]
