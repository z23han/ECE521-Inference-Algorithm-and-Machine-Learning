import numpy as np
import matplotlib.pyplot as plt


data = np.array([])
cluster_bucket = []
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
        cluster_bucket[i].append(indices[i])
    return


# calculate euclidean distance
def euclidean_dist():
    global e_dist
    for i in range(len(cluster_bucket)):
        dist = np.sum((data[cluster_bucket[i]] - data)**2.0, axis=1)
        e_dist.append(dist)
    return  


# update clusters, add points to cluster_bucket
def update_clusters():
    global cluster_bucket
    # reshape e_dist for calculations
    np_e_dist = np.array(e_dist)
    np_e_dist = np_e_dist.T
    for i in range(len(np_e_dist)):
        smallest_index = np.argsort(np_e_dist[i])[0]
        cluster_bucket[smallest_index].append(data[i])
    return


# plot a cluster
def plot_a_cluster(index, color):
    cluster = cluster_bucket[index][1:]
    # obtain x and y coordinates of the cluster
    x_cluster = np.array(cluster).T[0]
    y_cluster = np.array(cluster).T[1]
    plt.plot(x_cluster, y_cluster, color+'o')
    return 


# plot all the clusters
def plot_clusters():
    plt.figure(1)
    plot_a_cluster(0, 'r')
    plot_a_cluster(1, 'g')
    plot_a_cluster(2, 'b')
    plt.show()



def main(K):
    inputFile = 'data2D.npy'
    init_data(inputFile)
    init_centroids(K)
    euclidean_dist()
    update_clusters()
    plot_clusters()


if __name__ == '__main__':
    main(3)

