import numpy as np
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes
from sklearn.ensemble import RandomForestClassifier


class Stacking():

    def __init__(self):
        pass

    def fit(self, X, y):
        N,D = X.shape
        randomForestModel = RandomForest(num_trees=15, max_depth=np.inf)
        randomForestModel.fit(X,y)
        rf_predicted = randomForestModel.predict(X)

        naiveBayesModel = NaiveBayes()
        naiveBayesModel.fit(X,y)
        nb_predicted = naiveBayesModel.predict(X)

        knnModel = KNN(3)
        knn.fit(X, y)
        knn_predicted = knnModel.predict(X)

        self.rf_predicted = rf_predicted
        self.nb_predicted = nb_predicted
        self.knn_predicted = knn_predicted

        self.y = y

    def predict(self, X):
        y = self.y
        N, D = X.shape
        rf_predicted = self.rf_predicted
        nb_predicted = self.nb_predicted
        knn_predicted = self.knn_predicted

        stacked_predicted = (np.vstack((rf_predicted, nb_predicted, knn_predicted))).T

        decisionTreemodel = DecisionTree(max_depth=2)
        decisionTreemodel.fit(stacked_predicted, y)
        y_pred = decisionTreemodel.predict(stacked_predicted)

        return y_pred
