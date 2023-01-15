"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Homework 3
Clustering Classes
"""

import numpy as np
from scipy import stats

class Point(object):
    """
    Represents a data point
    """

    def __init__(self, features, label=None):
        """
        Initialize label and attributes
        """
        self.features = features
        self.label = label

    def dimensionality(self):
        """Returns dimension of the point"""
        return len(self.features)

    def get_features(self):
        """Returns features"""
        return self.features

    def distance(self, other):
        """
        other: point, to which we are measuring distance to
        Return Euclidean distance of this point with other
        """
        # TODO: Implement this function
        return np.linalg.norm(self.features - other.features)


    def get_label(self):
        """Returns label"""
        return self.label

class Cluster(object):
    """
    A Cluster is defined as a set of elements
    """

    def __init__(self, points):
        """
        Elements of a cluster are saved in a list, self.points
        """
        self.points = points

    def get_points(self):
        """Returns points in the cluster as a list"""
        return self.points

    def get_label(self):
        """Returns label of the cluster, which is determined by the
            mode of labels"""
        labels = [point.get_label() for point in self.points]
        cluster_label, count = stats.mode(labels)
        return cluster_label[0]

    def get_purity(self):
        """Returns number of points in cluster and the number of points
            with the most common label"""
        labels = [point.get_label() for point in self.points]
        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)

    def get_centroid(self):
        """Returns centroid of the cluster"""
        # TODO: Implement this function

        features = [point.get_features() for point in self.points]
        return np.sum(features, axis = 0) / len(features)


    def equivalent(self, other):
        """
        other: Cluster, what we are comparing this Cluster to
        Returns true if both Clusters are equivalent, or false otherwise
        """
        if len(self.get_points()) != len(other.get_points()):
            return False
        matched = []
        for p1 in self.get_points():
            for point2 in other.get_points():
                if p1.distance(point2) == 0 and point2 not in matched:
                    matched.append(point2)
        return len(matched) == len(self.get_points())

class ClusterSet(object):
    """
    A ClusterSet is defined as a list of clusters
    """

    def __init__(self):
        """
        Initialize an empty set, without any clusters
        """
        self.clusters = []

    def add(self, c):
        """
        c: Cluster
        Appends a cluster c to the end of the cluster list
        only if it doesn't already exist in the ClusterSet.
        If it is already in self.clusters, raise a ValueError
        """
        if c in self.clusters:
            raise ValueError
        self.clusters.append(c)

    def get_clusters(self):
        """Returns clusters in the ClusterSet"""
        return self.clusters[:]

    def get_centroids(self):
        """Returns centroids of each cluster in the ClusterSet as a list"""
        # TODO: Implement this function
        centroids = [cluster.get_centroid() for cluster in self.clusters]
        return centroids

    def get_score(self):
        """
            Returns accuracy of the clustering given by the clusters
            in ClusterSet object
        """
        total_correct = 0
        total = 0
        for c in self.clusters:
            n, n_correct = c.get_purity()
            total = total + n
            total_correct = total_correct + n_correct

        return total_correct / float(total)

    def num_clusters(self):
        """Returns number of clusters in the ClusterSet"""
        return len(self.clusters)

    def equivalent(self, other):
        """
        other: another ClusterSet object
        Returns true if both ClusterSets are equivalent, or false otherwise
        """
        if len(self.get_clusters()) != len(other.get_clusters()):
            return False
        matched = []
        for c1 in self.get_clusters():
            for c2 in other.get_clusters():
                if c1.equivalent(c2) and c2 not in matched:
                    matched.append(c2)
        return len(matched) == len(self.get_clusters())
