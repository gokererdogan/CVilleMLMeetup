#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.autograd as ag
from mxnet.io import NDArrayIter
import mxnet.ndarray as nd
import numpy as np
import tqdm


def get_data(batch_size):
    """
    Load MNIST data (training and test sets)
    """
    mnist_data = mx.test_utils.get_mnist()

    train_iter = NDArrayIter(mnist_data['train_data'], mnist_data['train_label'], batch_size, shuffle=True)
    test_iter = NDArrayIter(mnist_data['test_data'], mnist_data['test_label'], batch_size)

    return train_iter, test_iter


def get_minibatch(data_iter):
    try:
        batch = data_iter.next()
    except StopIteration:
        data_iter.reset()
        batch = data_iter.next()

    x = batch.data[0]
    x = nd.reshape(x, (x.shape[0], -1))
    y = nd.one_hot(batch.label[0], 10)
    return x, y


def predict(x, w, b):
    """
    Predict y given x and weight w, bias b.
    """
    z = nd.dot(x, w) + b
    return nd.exp(z) / nd.sum(nd.exp(z), axis=1, keepdims=True)  # softmax


def cross_entropy_loss(pred, true):
    """
    Cross entropy loss given predictions and true labels.
    """
    return nd.mean(nd.sum(-(nd.log(pred)*true), axis=1))


def calculate_accuracy(x, y, w, b):
    """
    Given x, y and parameters (w, b), calculate accuracy and cross entropy loss.
    """
    pred_y = predict(x, w, b)
    pred_labels = nd.argmax(pred_y, axis=1)
    true_labels = nd.argmax(y, axis=1)
    return nd.mean(pred_labels == true_labels).asscalar(), cross_entropy_loss(pred_y, y).asscalar()


def save_loss(loss_list, i, loss):
    if len(loss_list) == 0:
        loss_list.append([i, loss])
    else:
        loss_list.append([i, 0.9*loss + 0.1*loss_list[-1][1]])


def plot_prediction(batch_x, batch_y, i, w, b):
    """
    Given a batch (of x, y) and an index i, plot image and predictions.
    """
    xi = batch_x[i:(i+1)]
    yi = batch_y[i:(i+1)]
    img = xi.asnumpy().reshape((28, 28))
    plt.imshow(img, cmap='gray')
    pred_label = nd.argmax(predict(xi, w, b), axis=1).asscalar()
    true_label = nd.argmax(yi, axis=1).asscalar()
    plt.title("Prediction: {}, True: {}".format(pred_label, true_label))
    plt.show()
    pass


if __name__ == "__main__":
    input_dim = 28*28
    output_dim = 10
    learning_rate = 0.005
    batch_size = 128
    num_train_updates = 20000
    test_freq = 100  # measure accuracy and loss on test set every test_freq updates.
    num_test_batches = 10000 // 128  # use all test data

    # load data
    train_set, test_set = get_data(batch_size)

    # initialize parameters (weight and bias)
    w = nd.random_normal(shape=(input_dim, output_dim)) * 0.01
    b = nd.zeros(shape=(output_dim,))

    # tell mxnet to calculate gradients wrt to these variables.
    w.attach_grad()
    b.attach_grad()

    train_losses = []
    test_losses = []
    test_accs = []

    # plot prediction before training
    test_x, test_y = get_minibatch(test_set)
    plot_prediction(test_x, test_y, 0, w, b)

    pbar = tqdm.tqdm(total=num_train_updates)
    for i in range(num_train_updates):
        # get next training batch
        x, y = get_minibatch(train_set)
        # calculate prediction and loss
        with ag.record():  # this tells mxnet to record the operations below, so it can calculate the gradients
            # predict y
            pred_y = predict(x, w, b)
            # calculate loss
            loss = cross_entropy_loss(pred_y, y)

            save_loss(train_losses, i, loss.asscalar())
            pbar.set_postfix({"Train Loss": train_losses[-1][1]})
            pbar.update()

        loss.backward()  # calculate gradients

        # update parameters. gradient descent update.
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        if (i+1) % test_freq == 0:  # run evaluation on test set.
            accs = []
            losses = []
            for j in range(num_test_batches):
                x, y = get_minibatch(test_set)
                test_acc, test_loss = calculate_accuracy(x, y, w, b)
                accs.append(test_acc)
                losses.append(test_loss)
            save_loss(test_losses, i, np.mean(losses))
            save_loss(test_accs, i, np.mean(accs))

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    test_accs = np.array(test_accs)

    # plot training and test loss curves.
    plt.plot(train_losses[:, 0], train_losses[:, 1], label='train')
    plt.plot(test_losses[:, 0], test_losses[:, 1], label='test')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # plot test accuracy.
    plt.plot(test_accs[:, 0], test_accs[:, 1])
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

    # plot prediction after training.
    plot_prediction(test_x, test_y, 0, w, b)

    pass

