import numpy as np
import matplotlib.pyplot as plt


# data is a numpy array holding input data
data = np.array([])
# cluster_bucket is a list, whose first item is the cluster center, and the rest are cluster points
cluster_bucket = []
# e_dist is a list, holding all the distance between cluster center and cluster points
e_dist = []


# init data
def init_data(inputFile):
    global data
    data = np.load(inputFile)
    return


# init centroids
def init_centroids(K):
    global cluster_bucket
    # initialize cluster_bucket
    cluster_bucket = [[] for _ in range(K)]
    data_length = len(data)
    indices = range(data_length)
    # shuffle the indices to return K random values
    np.random.shuffle(indices)
    for i in range(K):
        cluster_bucket[i].append(data[indices[i]])
    return


# calculate euclidean distance
def euclidean_dist():
    global e_dist, cluster_bucket
    # clear e_dist if e_dist is not empty
    if e_dist != []:
        e_dist = []
    # refactor cluster_bucket if it doesn't contain only 1 element
    if len(cluster_bucket[0]) != 1:
        for i in range(len(cluster_bucket)):
            cluster_bucket[i] = [cluster_bucket[i][0]]
    # add euclidean distances to the list
    for i in range(len(cluster_bucket)):
        dist = np.sum((cluster_bucket[i] - data)**2.0, axis=1)
        e_dist.append(dist)
    return  


# update clusters, add points to cluster_bucket or update points in the cluster_bucket
def update_clusters():
    global cluster_bucket
    # reshape e_dist for calculations
    np_e_dist = np.array(e_dist)
    np_e_dist = np_e_dist.T
    
    for i in range(len(np_e_dist)):
        smallest_index = np.argsort(np_e_dist[i])[0]
        cluster_bucket[smallest_index].append(data[i])
    return


# update centroids by getting the median points among the each cluster
def update_centroids():
    global cluster_bucket
    for i in range(len(cluster_bucket)):
        center, cluster = cluster_bucket[i][0], cluster_bucket[i][1:]
        cluster = np.array(cluster)
        x_cluster = cluster.T[0]
        y_cluster = cluster.T[1]
        ave_x = sum(x_cluster)/float(len(x_cluster))
        ave_y = sum(y_cluster)/float(len(y_cluster))
        # update cluster centers
        cluster_bucket[i][0] = np.array([ave_x, ave_y])
    return


# do the iteration
def kmeans_clustering():
    K = 5
    max_iter = 500
    inputFile = 'data2D.npy'
    init_data(inputFile)
    init_centroids(K)
    euclidean_dist()
    update_clusters()
    print('cluster bucket')
    for i in range(K):
        print(cluster_bucket[i][0])
    plot_clusters(K)
    for _ in range(max_iter):
        update_centroids()
        euclidean_dist()
        update_clusters()
    print('cluster bucket')
    for i in range(K):
        print(cluster_bucket[i][0])
    plot_clusters(K)


# plot a cluster
def plot_a_cluster(index, color):
    cluster = cluster_bucket[index][1:]
    # obtain x and y coordinates of the cluster
    x_cluster = np.array(cluster).T[0]
    y_cluster = np.array(cluster).T[1]
    plt.plot(x_cluster, y_cluster, color+'o')
    return 


# plot all the clusters
def plot_clusters(K):
    plt.figure(1)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(K):
        plot_a_cluster(i, colors[i])
    plt.show()




if __name__ == '__main__':
    kmeans_clustering()

