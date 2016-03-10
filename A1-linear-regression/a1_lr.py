# a1: linear regression

import numpy as np
import matplotlib.pyplot as plt
import random


#### create training set
def createTrainingSet():
    train_x = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    train_y = np.sin(train_x) + 0.1 * np.power(train_x, 2) + 0.5 * np.random.randn(100,1)
    return train_x, train_y


#### linear fit
def linearFit(trainingSetX, trainingSetY):
    assert len(trainingSetX) == len(trainingSetY)
    omega = np.array([np.random.random(), np.random.random()])
    eta = 0.01
    T = len(trainingSetX)
    count = 0

    while True:
        y = []
        for t in range(T):
            yt = omega[0] + omega[1]*trainingSetX[t]
            y.append(yt)
        y = np.array(y)
        delta0 = 2./T * sum(trainingSetY - y)
        delta1 = 2./T * sum((trainingSetY - y)*trainingSetX)
        delta = np.array([delta0[0], delta1[0]])
        omega += eta * delta

        count += 1
        if count > 10000:
            break
        if sum(abs(delta)) < 0.00001:
            break
    print 'count', count
    print 'omega', omega
    print 'delta', delta
    return omega

#### normalize feature space
def normalizeInstance(instance):
    mean = np.mean(instance)
    std_dev = np.std(instance)
    return [(p-mean)/std_dev for p in instance]



#### polynomial fit
def polyFit(trainingSetX, trainingSetY, N):
    assert len(trainingSetX) == len(trainingSetY)
    trainingDataX = trainingSetX.flatten()
    trainingDataY = trainingSetY.flatten()

    ## store feature space, data structure is list
    featureSpace = []
    featureSpace.append([1.]*len(trainingDataX))
    for n in range(1, N):
        featureSpace.append(normalizeInstance(trainingDataX**n))
    ## initialize parameters
    omega = np.array([random.random() for _ in range(N)])
    eta = 0.01
    T = len(trainingDataX)
    count = 0

    while True:
        ## store y prediction
        y = np.zeros(T)
        for n in range(N):
            y += omega[n] * np.array(featureSpace[n])

        delta = []
        for n in range(N):
            di = 2./T * sum((trainingDataY - y)* np.array(featureSpace[n]))
            delta.append(di)

        delta = np.array(delta)
        omega += delta * eta

        count += 1
        if count > 30000:
            break
        if sum(abs(delta)) < 0.001:
            break


    print 'count', count
    print 'omega', omega
    print 'delta', delta

    return omega




#### plot the training data and fitted line
def dataPlot(trainingSetX, trainingSetY, omega, normalized=False):
    assert len(trainingSetX) == len(trainingSetY)
    trainingFlatten = trainingSetX.flatten()
    fittedY = np.zeros(len(trainingFlatten))
    for i in range(len(omega)):
        if i == 0:
            fittedY += np.array([omega[0] for _ in range(len(trainingFlatten))])
        else:
            if not normalized:
                fittedY += omega[i] * trainingFlatten**i
            else:
                fittedY += omega[i] * np.array(normalizeInstance(trainingFlatten**i))

    line1, = plt.plot(trainingSetX, trainingSetY, 'ro')
    line2, = plt.plot(trainingSetX, fittedY, 'g')
    plt.legend(["training data", "fitted line"])
    plt.show()


#### task 3
def task3():
    train_x, train_y = createTrainingSet()
    omega = linearFit(train_x, train_y)
    dataPlot(train_x, train_y, omega)

    #for mean square error calculation
    x_train = np.empty((len(train_x),2))
    x_train[:,0] = 1
    x_train[:,1] = train_x.T
    pred = np.dot(x_train, omega.T)
    cost = 0.5 * np.mean((pred - train_y)**2)
    print cost

#### task 4
def task4():
    train_x, train_y = createTrainingSet()
    omega = polyFit(train_x, train_y, 6)
    dataPlot(train_x, train_y, omega, normalized=True)



if __name__ == '__main__':
    task4()




