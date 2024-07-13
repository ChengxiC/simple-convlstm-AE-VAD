import torch
from matplotlib import pyplot as plt
import numpy as np


def plot_loss(epochs, loss, plot=True):
    """

    :param epochs: the number of training iterations
    :param loss: train loss, a list
    :return: epochs vs loss
    """
    if plot is False:
        return

    if epochs != len(loss):
        print('the number of epochs does not match the length of the loss list')
        return

    x = np.arange(1, epochs+1)
    plt.plot(x, loss, label='loss_epoch', color='b', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss over epochs')
    plt.legend()
    plt.show()


def plot_scores(frames, scores, plot=True):

    if plot is False:
        return

    if frames != len(scores):
        print('the number of epochs does not match the length of the loss list')
        return

    x = np.arange(1, frames+1)
    plt.plot(x, scores, label='scores', color='b', linewidth=1)
    plt.xlabel('frames')
    plt.ylabel('score')
    plt.title('anomaly scores')
    plt.legend()
    plt.show()




