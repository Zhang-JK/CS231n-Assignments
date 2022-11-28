import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    for i in range(num_train):
        score = np.dot(X[i], W).T
        # score -= np.max(score)
        exp_sum = np.sum(np.exp(score))
        loss += -np.log(np.exp(score[y[i]]) / exp_sum)
        for j in range(W.shape[1]):
            if j != y[i]:
                dW[:, j] += X[i] * np.exp(score[j]) / exp_sum
            else:
                dW[:, j] += X[i] * (np.exp(score[j]) / exp_sum - 1)

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    score = np.exp(np.dot(X, W))
    exp_sum = np.sum(score, axis=1)
    loss = np.average(-np.log(score[np.arange(num_train), y] / exp_sum)) + reg * np.sum(W * W)
    weight = (score.T / exp_sum)
    weight[y, np.arange(num_train)] -= 1
    dW = np.dot(weight, X).T / num_train + 2 * reg * W
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
