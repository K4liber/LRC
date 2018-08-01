from sklearn.neighbors import NearestNeighbors
import operator
import numpy as np

class KNN:
    nn = None
    targets = None
    learn_data = None

    def __init__(self, k, algoritm, learn_data, targets):
        self.nn = NearestNeighbors(n_neighbors=k, algorithm=algoritm).fit(learn_data)
        self.targets = targets
        self.learn_data = learn_data

    def getResponse(self, neighbors):
        classVotes = {}

        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1

        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        return sortedVotes[0][0]

    def predict(self, predict_data):
        predictions = self.nn.kneighbors(predict_data, return_distance=False)
        predicted_classes = [0 for x in range(len(predictions))]

        for j in range(len(predictions)):
            predicted_classes[j] = self.getResponse(self.targets[predictions[j]])
        
        return predicted_classes

    def predict2(self, predict_data):
        nn = self.nn.kneighbors(predict_data, return_distance=False)
        predicted_classes = [0 for x in range(len(predict_data))]
        bin_threshold = sum(self.targets)/len(self.targets)
 
        for j in range(len(predict_data)):
            
            weight = [0 for x in range(len(nn[j]))]
            prediction = 0.0

            for i in range(len(weight)):
                index = nn[j][i]
                weight[i] = 1.0/(np.linalg.norm(self.learn_data[index]-predict_data[j]))
                prediction += weight[i] * self.targets[index]

            norm = sum(weight)
            prediction = prediction/norm    

            if prediction > bin_threshold:
                predicted_classes[j] = 1               

        return predicted_classes