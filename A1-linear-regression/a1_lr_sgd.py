# a1: linear regression using stochastic gradient descent

import numpy as np
import matplotlib.pyplot as plt


#### load dataset
def loadDataSet():
    with np.load("TINY_MNIST.npz") as data:
        x, t = data["x"], data["t"]
        x_eval, t_eval = data["x_eval"], data["t_eval"]
    return x, t, x_eval, t_eval



#### pick a mini-batch from training cases
def pickMiniBatch(instance1, instance2, M):
    assert len(instance1) == len(instance2)
    assert len(instance1) >= M
    inst1FirstM = list(instance1[:M])
    inst2FirstM = list(instance2[:M])
    return inst1FirstM, inst2FirstM


#### shuffle training data
def stochasticShuffle(instance1, instance2):
    assert len(instance1) == len(instance2)
    index_shuf = range(len(instance1))
    np.random.shuffle(index_shuf)
    inst1_shuf = [instance1[i] for i in index_shuf]
    inst2_shuf = [instance2[i] for i in index_shuf]
    return inst1_shuf, inst2_shuf



#### calculate errors
def calValidationErrors(t_eval, t_pred):
    assert len(t_eval) == len(t_pred)
    num_of_errors = 0
    for i in range(len(t_pred)):
        if t_pred[i] < 0.5:
            t_val = 0
        else:
            t_val = 1
        if t_val != t_eval[i][0]:
            num_of_errors += 1
    return num_of_errors


#### calculate t prediction
def calPrediction(testingSetX, omega, b):
    t_pred = []
    for i in range(len(testingSetX)):
        t_pred.append(sum(testingSetX[i]*omega) + b)
    return np.array(t_pred)



##### steepest descent method
def steepestDescent(trainingSetX, trainingSetT, N, miniBatchSize=50, isMiniBatch=False, epoch=1000):

    trainingSetXX = trainingSetX[:N]
    trainingSetTT = trainingSetT[:N].flatten()
    ### initialization of omega, with size of 64
    omega = [np.random.random() for _ in range(64)]
    omega = np.array(omega)
    ### initialization of b
    b = np.random.random()

    eta = 0.01
    count = 0

    ## shuffle the training data
    trainingDataX, trainingDataT = stochasticShuffle(trainingSetXX, trainingSetTT)
    trainingDataX = np.array(trainingDataX)
    trainingDataT = np.array(trainingDataT)

    if isMiniBatch:
        epoch = epoch * N / miniBatchSize

        while True:
            left = N
            while left > 0:
                ## initialize prediction of y
                y = []
                trainingDataX = trainingDataX[:miniBatchSize]
                trainingDataT = trainingDataT[:miniBatchSize]
                for t in range(miniBatchSize):
                    yt1 = sum(omega*trainingDataX[t])
                    yt2 = b
                    yt = yt1 + yt2
                    y.append(yt)

                y = np.array(y)
                delta0 = 1./miniBatchSize* sum(trainingDataT - y)
                delta1 = [1./miniBatchSize* sum((trainingDataT-y)*trainingDataX[:,i]) for i in range(64)]

                b = b + eta * delta0
                omega = omega + eta * np.array(delta1)

                left = left - miniBatchSize

            count += 1
            if count > epoch:
                break
            if sum(abs(np.array(delta1))) < 0.001:
                break

        return omega, b

    else:
        epoch = 100000
        omegas = []
        bs = []
        counts = []
        cnt = 0
        while True:

            ## initialize prediction of y
            y = []
            for t in range(N):
                dataX = trainingDataX[t]
                yt1 = sum(omega*dataX)
                yt2 = b
                yt = yt1 + yt2
                y.append(yt)
            y = np.array(y)
            delta0 = 1./N* sum(trainingDataT - y)
            delta1 = [1./N* sum((trainingDataT-y)*trainingDataX[:,i]) for i in range(64)]

            b = b + eta * delta0
            omega = omega + eta * np.array(delta1)

            cnt += 1
            count += 1
            if cnt == 10:
                omegas.append(omega)
                bs.append(b)
                counts.append(count)
                cnt = 0


            if count > epoch:
                break
            if sum(abs(np.array(delta1))) < 0.001:
                break

        return omegas, bs, counts



#### linear regression with regularization
def lrRegularization(trainingSetX, trainingSetT, regLambda=0.0001, N=50):
    assert len(trainingSetX) == len(trainingSetT)
    trainingSetXX = trainingSetX[:N]
    trainingSetTT = trainingSetT[:N].flatten()

    ## initialize omega
    omega = [np.random.random() for _ in range(64)]
    omega = np.array(omega)
    ### initialization of b
    b = np.random.random()

    eta = 0.01
    count = 0

    ## shuffle the training data
    trainingDataX, trainingDataT = stochasticShuffle(trainingSetXX, trainingSetTT)
    trainingDataX = np.array(trainingDataX)
    trainingDataT = np.array(trainingDataT)

    while True:

        ## initialize prediction of y
        y = []
        for t in range(N):
            dataX = trainingDataX[t]
            yt1 = sum(omega*dataX)
            yt2 = b
            yt = yt1 + yt2
            y.append(yt)
        y = np.array(y)

        delta0 = 1./N* sum(trainingDataT - y) - regLambda * b
        delta1 = np.array([1./N* sum((trainingDataT-y)*trainingDataX[:,i]) for i in range(64)]) - \
                 regLambda * omega

        b = b + eta * delta0
        omega = omega + eta * np.array(delta1)

        count += 1
        if count > 20000:
            break
        if sum(abs(np.array(delta1))) < 0.0001:
            break

    return omega, b



### plot training errors and validation errors vs. epochs
def errorPlot(trainingSetX, trainingSetT, testingSetX, testingSetT,
                  omegas, bs, counts):
    assert len(trainingSetX) == len(trainingSetT)
    assert len(testingSetX) == len(testingSetT)
    trainingErrors = []
    testingErrors = []
    for i in range(len(counts)):
        t_train_pred = calPrediction(trainingSetX, omegas[i], bs[i])
        train_error = calValidationErrors(t_eval=trainingSetT, t_pred=t_train_pred)
        trainingErrors.append(train_error)
        t_test_pred = calPrediction(testingSetX, omegas[i], bs[i])
        test_error = calValidationErrors(t_eval=testingSetT, t_pred=t_test_pred)
        testingErrors.append(test_error)
    line1, = plt.plot(counts, trainingErrors, 'r')
    line2, = plt.plot(counts, testingErrors, 'g')
    plt.xlabel('epoch')
    plt.ylabel('errors')
    plt.legend(['Training Errors', 'Testing Errors'], loc='upper center')
    plt.show()





#### task 5 alter N points
def task5AlterNPoints():
    N = [100, 200, 400, 800]
    x, t, x_eval, t_eval = loadDataSet()
    errors = []
    percentages = []
    for n in N:
        omega, b = steepestDescent(x, t, n, miniBatchSize=50, isMiniBatch=True)
        t_pred = calPrediction(x_eval, omega, b)
        error = calValidationErrors(t_eval, t_pred)
        percentage = 1 - error/float(400)
        errors.append(error)
        percentages.append(percentage)
    return errors, percentages


#### task 6 over-fitting, alter the number of epoch
def task6AlterEpoch(N=50):
    x, t, x_eval, t_eval = loadDataSet()
    omegas, bs, counts = steepestDescent(x, t, N, miniBatchSize=50, isMiniBatch=False)
    errorPlot(x, t, x_eval, t_eval, omegas, bs, counts)



#### task 7 regularization altering lambda
def task7Regularization():
    regLambda = [0, 0.0001, 0.001,0.01, 0.1, 0.5]
    x, t, x_eval, t_eval = loadDataSet()
    errors = []
    percentages = []
    for l in regLambda:
        omega, b = lrRegularization(x, t, regLambda=l)
        t_pred = calPrediction(x_eval, omega, b)
        error = calValidationErrors(t_eval, t_pred)
        percentage = 1 - error/float(400)
        errors.append(error)
        percentages.append(percentage)
    return errors, percentages



### task 5
def task5():
    print task5AlterNPoints()


### task 6
def task6():
    print task6AlterEpoch()

#### task 7
def task7():
    print task7Regularization()


if __name__ == '__main__':
    task7()