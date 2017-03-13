import numpy as np
from random import randrange
import csv

def predict(row, coefficients):
    sum = coefficients[0]
    for i in range(len(row) - 1):
        sum += coefficients[i+1] * float(row[i])
    #print 1.0 / (1.0 + np.exp(-sum))
    return 1.0 / (1.0 + np.exp(-sum))


def stochastic_gradient_descent(train, learn_rate, num_epochs):
    coef = [0.0] * (len(train[0]))
    for epoch in range(num_epochs):
        # print epoch
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
    # print "gradient descent"
    coef = stochastic_gradient_descent(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def log_reg_to_historical_csv(train, test, l_rate, n_epoch, season):
    csv_rows = []
    minmax = dataset_minmax(train)
    normalize_dataset(train, minmax)
    coef = stochastic_gradient_descent(train, l_rate, n_epoch)
    for row in test:
        csv_row = []
        csv_row.append(row.pop(0)) #team1
        csv_row.append(row.pop(0)) #team2
        csv_row.append(row.pop(0)) #season
        csv_row.append(predict(row, coef)) #prediction
        csv_row.append(row[-1]) #actual
        csv_rows.append(csv_row)
    with open('data/historical_csv' + season + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)

def final_submission(train, test, l_rate, n_epoch):
    csv_rows = []
    minmax = dataset_minmax(train)
    normalize_dataset(train, minmax)
    coef = stochastic_gradient_descent(train, l_rate, n_epoch)
    for row in test:
        csv_row = []
        csv_row.append((row.pop(0))) #team1
        csv_row.append(row.pop(0)) #team2
        csv_row.append(predict(row, coef))
        csv_rows.append(csv_row)
    with open('data/final_predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)



