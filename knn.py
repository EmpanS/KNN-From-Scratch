import numpy as np

class KNN:
    def __init__(self, data, k):
        """A class for the supervised learning-algorithm KNN. Implemented using only numpy. KNN is a lazy learner meaning that all calculations is done when predicting new observations.
        
        Attributes
        ----------
        data : (numpy array)
            A numpy array containing all training data for the model. The last column should contain the labels. 
            
        k : (integer)
            The hyper-parameter k, specifies how many neighbors to take into account when classifying observations.
            
        n : (integer)
            Number of observations.
        """

        self.data = data
        self.k = k
        self.n = data.shape[0]
        
    def predict(self, new_points):
        """Predict the classification of new points.
        
        Parameters
        ----------
        new_points : (numpy array)
            An array of arrays containing all the points to predict. If only one point is to be predicted, it still has to be an array of array:
                                   e.g. new_points = np.array([[feature 1, ..., feature 2]])
        
        Returns
        -------
        predictions : (numpy array) 
        """

        predictions = np.zeros(new_points.shape[0])
        # Loop over each new point to classify
        for i, point in enumerate(new_points): 
            distances = self._calculate_distances(point)

            # Finds the k smallest distances and the corresponding points' labels
            label_neighbors = self.data[:,-1][np.argpartition(distances, self.k)[:self.k]]

            # Finds the majority of k nearest neighbors
            predictions[i] = np.bincount(label_neighbors.astype('int64')).argmax()

        return predictions
        
    def _calculate_distances(self, new_point):
        """Calculate the euclidean distance between a new point and all points in self.data
        
        Parameters
        ----------
        new_points : (numpy array)
            An array of containing the new point. predict.
        
        Returns
        -------
        euclidean_distance : (numpy array) 
            An array containing the distances between all the new point and all points in the model.
        """

        # Expand by repeating to increase speed (avoid looping) 
        new_point = np.resize(new_point, (self.n, new_point.shape[0]))
        euclidean_distance = np.sum((self.data[:,0:-1] - new_point)**2, axis=1)
        return euclidean_distance