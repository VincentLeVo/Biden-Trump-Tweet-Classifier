
import numpy as np
import utils
from kmeans import Kmeans
from utils import *
from scipy import stats


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = mode(y[X[:, d] > value])
                y_not = mode(y[X[:, d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat


"""
A helper function that computes the Gini_impurity of the
discrete distribution p.
    """
def Gini_impurity(p):
        # pass
        #Finished
    "Insert your code here "
    return np.sum(p*(1-p))


class DecisionStumpGiniIndex(DecisionStumpErrorRate):
    #Question

    def fit(self, X, y, split_features=None, thresholds=None):
        # you may get RuntimeWarning: invalid value encountered Warning but that's ok.
        # Also, it may takes between 5 to 10 min to see the result.
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = None
        self.splitNot = None
        self.splitVariable = y_mode
        self.splitValue = None

        #Array of probailities and plug into Gini Impurity
        p = count/np.sum(count)
        #This is the minimum error so far.
        giniIndexSoFar = Gini_impurity(p)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split and their counts/total
                y_R = mode(y[X[:, d] > value])
                n_R = 1 # Base case for Right counts
                n_R = np.bincount(y_R)

                y_L = mode(y[X[:, d] <= value])
                n_L= 1 # Base case for Left counts
                n_L = np.bincount(y_L)

                #Total counts
                n_Total = n_R + n_L

                #Data
                data_Total = np.sum(n_Total)
                data_R = np.sum(n_R)
                data_L = np.sum(n_L)

                #probability of the right side and left side
                p_R = n_R/data_R
                p_L = n_L/data_L

                # Compute Gini Index
                giniIndex = ((data_L/data_Total)*Gini_impurity(p_L))+((data_R/data_Total)*Gini_impurity(p_R))

                # Compare to minimum error so far
                if giniIndex < giniIndexSoFar:
                    # This is the lowest error, store this value
                    minGiniIndex = giniIndex
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = np.argmax(n_R)
                    self.splitNot = np.argmax(n_L)


"""**Decision Tree**"""


class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class

    def fit(self, X, y, thresholds=None):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape

        # Learn a decision stump
        splitModel = self.stump_class()
        splitModel.fit(X, y, thresholds=thresholds)

        if self.max_depth <= 1 or splitModel.splitVariable is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.splitModel = splitModel
            self.subModel1 = None
            self.subModel0 = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel.splitVariable
        value = splitModel.splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:, j] > value
        splitIndex0 = X[:, j] <= value

        # Fit decision tree to each split
        self.splitModel = splitModel
        self.subModel1 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel1.fit(X[splitIndex1], y[splitIndex1], thresholds=thresholds)
        self.subModel0 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel0.fit(X[splitIndex0], y[splitIndex0], thresholds=thresholds)

    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        # GET VALUES FROM MODEL
        splitVariable = self.splitModel.splitVariable
        splitValue = self.splitModel.splitValue
        splitSat = self.splitModel.splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        # the case with depth=1, just a single stump.
        elif self.subModel1 is None:
            return self.splitModel.predict(X)

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:, j] > value
            splitIndex0 = X[:, j] <= value

            y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self.subModel0.predict(X[splitIndex0])

        return y

class RandomStumpGiniIndex(DecisionStumpGiniIndex):

        def fit(self, X, y, thresholds=None):
            # Randomly select k features.
            # This can be done by randomly permuting
            # the feature indices and taking the first k
            D = X.shape[1]
            k = int(np.floor(np.sqrt(D)))

            chosen_features = np.random.choice(D, k, replace=False)

            DecisionStumpGiniIndex.fit(self, X, y, split_features=chosen_features, thresholds=thresholds)


"""**Random Tree**"""


class RandomTree(DecisionTree):

    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpGiniIndex)

    def fit(self, X, y, thresholds=None):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y, thresholds=thresholds)


"""**Random Forest**"""


class RandomForest:

    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.thresholds = None

    def fit(self, X, y):
        self.trees = []
        self.create_splits(X)
        for m in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y, thresholds=self.thresholds)
            self.trees.append(tree)

    def predict(self, X):
        t = X.shape[0]
        yhats = np.ones((t, self.num_trees), dtype=np.uint8)
        # Predict using each model
        for m in range(self.num_trees):
            yhats[:, m] = self.trees[m].predict(X)

        # Take the most common label
        return stats.mode(yhats, axis=1)[0].flatten()


    '''
    One way of implementing code for determining thresholds using K-means is to do it as method inside the random forest class,
    and send the thresholds you found for all the d features inside the self.threshold.
    '''
    def create_splits(self, X):
        N, D = X.shape
        kmeans = Kmeans(5)
        splits = np.zeros((D * kmeans.k,))
        for d in range(D):
            current = X[:,d]
            current = np.array(current, ndmin=2).T
            kmeans.fit(current)
            for j in range(kmeans.k):
                splits[j] = kmeans.means[j,]
        self.splits = splits
        return splits
