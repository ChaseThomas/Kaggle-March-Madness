import numpy as np
from random import randrange

def predict(row, coefficients):
    sum = coefficients[0]
    for i in range(len(row) - 1):
        sum += coefficients[i+1] * row[i]
    return 1.0 / (1.0 + np.exp(-sum))


def stochastic_gradient_descent(train, learn_rate, num_epochs):
    coef = [0.0] * (len(train[0]))
    for epoch in range(num_epochs):
        print epoch
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] += learn_rate * error * yhat * (1.0-yhat)
            for i in range(len(row) - 1):
                coef[i+1] += learn_rate * error * yhat * (1.0-yhat) * row[i]
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    print "gradient descent"
    coef = stochastic_gradient_descent(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions