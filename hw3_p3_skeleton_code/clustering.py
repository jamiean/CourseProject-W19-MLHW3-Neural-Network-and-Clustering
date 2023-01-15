"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Homework 3
Clustering
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from operator import methodcaller

from clustering_classes import Cluster, ClusterSet, Point
from data.landmarks import LandmarksDataset
from utils import denormalize_image

def random_init(points, k):
    """
    Arguments:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO: Implement this function
    points_feature = [point.get_features() for point in points]
    random_result = random.sample(range(0, len(points)), k)
    result = []
    for i in range(k) : result += [points_feature[random_result[i]]]
    return result

def k_means_pp_init(points, k):
    """
    Arguments:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points selected from points
    """
    # TODO: Implement this function
    points_feature = [point.get_features() for point in points]
    result = [points_feature[random.randint(0, len(points))]]

    while len(result) != k:
        distence = np.array([])
        for feature in points_feature:
            min_dis = float("inf")
            for centroid in result: min_dis = min(min_dis, feature.distence(centroid))
            distence = np.append(min_dis ** 2)
        distence = distence / np.max(distence)
        random_sel_num = np.random.choice(points_feature, 1, p = distence)
        result += [points_feature[np.random.choice(points_feature, 1, p = distence)[0]]]

    return result





def k_means(points, k, init='random'):
    """
    Clusters points into k clusters using k_means clustering.
    Arguments:
        points: a list of Point objects
        k: the number of clusters
        init: The method of initialization. One of ['random', 'kpp'].
              If init='kpp', use k_means_pp_init to initialize clusters.
              If init='random', use random_init to initialize clusters.
              Default value 'random'.
    Returns:
        Instance of ClusterSet with k clusters
    """
    # TODO: Implement this function
    centroids = []
    if init == 'random': centroids = random_init(points, k)
    if init == 'kpp': centroids = k_means_pp_init(points, k)
    points_feature = [point.get_features() for point in points]
    points_cluster = [[] for i in range(k)]

    for n in len(points_feature):
        min_dis, min_cluster = float("inf"), 0
        for i in range(len(centroids)): 
            if min_dis > points_feature[n].distence(centroid[i]): min_cluster = i
        points_cluster[i] += [points[n]]

    clusters = [Cluster(point) for point in points_cluster]
    Cluster_set = Clusterset()
    for cluster in clusters: cluster_sets.add(Cluster)
    Cluster_new = Clusterset()


    while (!Cluster_new.equivalent(Cluster_set)):
        Cluster_set = Cluster_new
        centroids = [cluster.get_centroid() for cluster in Cluster_set.get_clusters()]
        points_cluster = [[] for i in range(k)]

        for n in len(points_feature):
            min_dis, min_cluster = float("inf"), 0
            for i in range(len(centroids)): 
                if min_dis > points_feature[n].distence(centroid[i]): min_cluster = i
            points_cluster[i] += [points[n]]

        clusters = [Cluster(point) for point in points_cluster]
        Cluster_new = Clusterset()
        for cluster in clusters: cluster_new.add(Cluster)

    return Cluster_new;



def spectral_clustering(points, k):
    """
    Uses sklearn's spectral clustering implementation to cluster the input
    data into k clusters
    Arguments:
        points: a list of Points objects
        k: the number of clusters
    Returns:
        Instance of ClusterSet with k clusters
    """
    X = np.array([point.get_features() for point in points])
    spectral = SpectralClustering(
        n_clusters=k, n_init=1, affinity='nearest_neighbors', n_neighbors=50)
    y_pred = spectral.fit_predict(X)
    clusters = ClusterSet()
    for i in range(k):
        cluster_members = [p for j, p in enumerate(points) if y_pred[j] == i]
        clusters.add(Cluster(cluster_members))
    return clusters

def plot_performance(k_means_scores, kpp_scores, spec_scores, k_vals):
    """
    Uses matplotlib to generate a graph of performance vs. k
    Arguments:
        k_means_scores: A list of len(k_vals) average purity scores from
            running the k-means algorithm with random initialization
        kpp_scores: A list of len(k_vals) average purity scores from running
            the k-means algorithm with k_means++ initialization
        spec_scores: A list of len(k_vals) average purity scores from running
            the spectral clustering algorithm
        k_vals: A list of integer k values used to calculate the above scores
    """
    # TODO: Implement this function
    plt.ylabel("Purity")
    plt.plot(k_vals, k_means_scores, label = "k-means")
    plt.plot(k_vals, kpp_scores, label = "k-means++")
    plt.plot(k_vals, spec_scores, label = "spectral")
    plt.legend()
    plt.show()


def get_data():
    """
    Retrieves the data to be used for the k-means clustering as a list of
    Point objects
    """
    landmarks = LandmarksDataset(num_classes=5)
    X, y = landmarks.get_batch('train', batch_size=400)
    X = X.reshape((len(X), -1))
    return [Point(image, label) for image, label in zip(X, y)]

def visualize_clusters(kmeans, kpp, spectral):
    """
    Uses matplotlib to generate plots of representative images for each
    of the clustering algorithm. In each image, every row is from the same
    cluster, and from leftmost image is the medoid. Intra-cluster distance
    increases as we go from left to right.
    Arguments:
        - kmeans, kpp, and spectral: ClusterSet instances
    """
    def get_medoid_and_neighbors(points, num=4):
        D = pairwise_distances([p.features for p in points])
        distances = D.mean(axis=0)
        return np.array(points)[np.argsort(distances)[:num]].tolist()

    names = ['k-means', 'k-means++', 'spectral']
    cluster_sets = [kmeans, kpp, spectral]
    clusters_s = [sorted(cs.get_clusters(),
                  key=methodcaller('get_label')) for cs in cluster_sets]

    for i, clusters in enumerate(clusters_s):
        num = 4
        k = len(clusters)
        fig, axes = plt.subplots(nrows=k, ncols=num, figsize=(8,8))
        plt.suptitle(names[i])
        for j in range(k):
            pts = get_medoid_and_neighbors(clusters[j].get_points(), num)
            for n in range(len(pts)):
                axes[j,n].imshow(denormalize_image(
                np.reshape(pts[n].features,(32,32,3))), interpolation='bicubic')
            axes[j,0].set_ylabel(clusters[j].get_label())

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig('4j_clusters_viz_{}.png'.format(names[i]),
                    dpi=200, bbox_inches='tight')

def main():
    points = get_data()
    # TODO: Implement this function
    # for 3.h and 3.i

    """ 3.j """
    # Display representative examples of each cluster for clustering algorithms
    np.random.seed(42)
    kmeans = [k_means(points, i, 'random').get_score() for i in range(1,11)]
    kpp = [k_means(points, i, 'kpp').get_score() for i in range(1,11)]
    spectral = [spectral_clustering(points, i).get_score() for i in range(1,11)]
    visualize_clusters(kmeans, kpp, spectral)

if __name__ == '__main__':
    main()
