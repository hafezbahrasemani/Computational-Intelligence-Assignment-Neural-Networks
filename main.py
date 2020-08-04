import csv
import random

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def csv_reader(filename, data_holder, labels_holder):
    with open(filename) as file:
        for line in file:
            words = line.split(',')
            data_holder.append([float(words[0]), float(words[1])])
            labels_holder.append(int(words[2]))


def draw_data(X, y, X_test, y_test, confirmed):
    confirmed_x = []
    confirmed_y = []


    for i in range(len(confirmed)):
        confirmed_x.append(confirmed[i][0])
        confirmed_y.append(confirmed[i][1])

    plt.plot(X, y, 'bo', color='red')
    plt.plot(X_test, y_test, 'bo', color='blue')
    plt.plot(confirmed_x, confirmed_y, 'bo', color='green')
    plt.show()


def compute_y(W, X, b):
    return np.matmul(X, W) + b


def compute_cost(y, y0):
    return (1 / 2) * pow(y - y0, 2)


def calculate_derivative(X, y, W, b, y0):
    m = y.size

    z = -(X * W + b)

    derivative_based_W = X * (y - y0) * sigmoid_prime(z)
    derivative_based_b = (y - y0) * sigmoid_prime(z)

    return derivative_based_W, derivative_based_b


def learning(W, b, n_epoch, lr, training_data, test_data):
    n_w = len(W)
    n_train = len(training_data)
    n_test = len(test_data)

    for i in range(n_epoch):
        grad = np.zeros([len(W)])
        for j in range(n_w):
            for k in range(n_train):
                y = compute_y(W, training_data[k], b)
                cost = compute_cost(y, labels[k])

                # print(cost)

                derivative_based_W, derivative_based_b = calculate_derivative(training_data[k][j], y, W[j], b, labels[k])

                grad[j] += derivative_based_W
                # grad[j] += derivative_based_W[1]

        for j in range(n_w):
            W[j] -= lr * grad[j]

    confirmed = []

    # Test part
    for i in range(n_train, n_train + n_test):
        y = -(np.matmul(data[i], W) + b)
        difference = abs(y - labels[i])

        if difference < 0.25:
            print("Test Data {0}: {1} ---- Confirmed".format(i, y))
            confirmed.append(data[i])
        else:
            print("Test Data {0}: {1} ---- Lost".format(i, y))

    return confirmed

if __name__ == '__main__':
    data = []
    labels = []
    training_data = []
    test_data = []

    W = np.random.rand(2)
    b = np.random.rand(1)
    lr = 0.01
    n_epoch = 300

    csv_reader("data.csv", data, labels)

    X = []
    y = []

    X_test = []
    y_test = []

    confirmed = []

    for i in range(150):
        X.append(data[i][0])
        y.append(data[i][1])
        training_data.append(data[i])
    for i in range(150, 200):
        X_test.append(data[i][0])
        y_test.append(data[i][1])
        test_data.append(data[i])


    confirmed = learning(W, b, n_epoch, lr, training_data, test_data)

    draw_data(X, y, X_test, y_test, confirmed)
