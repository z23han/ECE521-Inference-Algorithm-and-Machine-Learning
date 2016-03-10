# a1: k-nearest neighbors

import numpy as np


#### load dataset
def loadDataSet():
    with np.load("TINY_MNIST.npz") as data:
        x, t = data["x"], data["t"]
        x_eval, t_eval = data["x_eval"], data["t_eval"]
    return x, t, x_eval, t_eval



#### calculate Euclidean distance
def calEuclideanDist(instance1, instance2):
    assert len(instance1) == len(instance2)
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i])**2
    return (float(distance))**0.5


#### alter trainingSet N and set K = 1
def knnTrainingSize(trainingSetX, testingSetX, trainingSetT, N):
    trainingDataX = np.array(trainingSetX[:N])
    trainingDataT = np.array(trainingSetT[:N])
    testingData = np.array(testingSetX)
    t_pred = []
    for i in range(len(testingData)):
        dist = []
        for j in range(N):
            dist.append(calEuclideanDist(testingData[i], trainingDataX[j]))
        order = np.argsort(np.array(dist))
        t = trainingDataT[order[0]]
        t_pred.append(t)
    return t_pred




#### justify whether first K is 1 or 0
def justifyFirstKs(t, firstK, K):
    lst = []
    for i in range(K):
        lst.append(t[firstK[i]])
    if sum(lst) > K/2:
        return 1
    else:
        return 0



#### alter K and set N = 800
def knnAlterK(trainingSetX, testingSetX, trainingSetT, k):
    trainingDataX = np.array(trainingSetX[:800])
    trainingDataT = np.array(trainingSetT[:800])
    testingData = np.array(testingSetX)
    t_pred = []
    for i in range(len(testingData)):
        dist = []
        for j in range(800):
            dist.append(calEuclideanDist(testingData[i], trainingDataX[j]))
        order = np.argsort(np.array(dist))
        firstK = order[:k]
        t = justifyFirstKs(trainingDataT, firstK, k)
        t_pred.append(t)
    return t_pred



#### calculate validation error
def calError(t_pred, t_eval):
    assert len(t_pred) == len(t_eval)
    error = 0
    N = len(t_pred)
    for i in range(N):
        error += abs(t_pred[i] - t_eval[i])
    error = 1./N * error
    return error



#### task 1 set N
def task1AlterTrainingSize():
    N = [5, 50, 100, 200, 400, 800]
    x, t, x_eval, t_eval = loadDataSet()
    errors = []
    for n in N:
        t_pred = knnTrainingSize(x, x_eval, t, n)
        error = calError(t_pred, t_eval)
        errors.append(error)
    for i in range(len(errors)):
        print N[i], errors[i]
    return errors


#### task 2 set k
def task2AlterK():
    K = [1, 3, 5, 7, 21, 101, 401]
    x, t, x_eval, t_eval = loadDataSet()
    errors = []
    for k in K:
        t_pred = knnAlterK(x, x_eval, t, k)
        error = calError(t_pred, t_eval)
        errors.append(error)
    for i in range(len(errors)):
        print K[i], errors[i]
    return errors



if __name__ == '__main__':
    #task1AlterTrainingSize()
    task2AlterK()





