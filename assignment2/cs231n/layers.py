from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # Implement the affine forward pass. Store the result in out. You         #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    mid = np.reshape(x, (x.shape[0], w.shape[0]))
    out = np.dot(mid, w) + b
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # Implement the affine backward pass.                                     #
    ###########################################################################
    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(np.reshape(x, (x.shape[0], w.shape[0])).T, dout)
    db = np.sum(dout, axis=0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # Implement the ReLU forward pass.                                        #
    ###########################################################################
    out = np.array(x, copy=True)
    out[out < 0] = 0
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # Implement the ReLU backward pass.                                       #
    ###########################################################################
    dx = dout
    dx[x < 0] = 0
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # Implement the training-time forward pass for batch norm.            #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        cache = {}
        xm = x.mean(axis=0)
        xv = x.var(axis=0)
        cache['std'] = np.sqrt(xv + eps)
        cache['norm'] = (x - xm) / cache['std']
        cache['gamma'] = gamma
        cache['x'] = x
        cache['m'] = xm
        out = (cache['norm'] * gamma) + beta
        running_mean = momentum * running_mean + (1 - momentum) * xm
        running_var = momentum * running_var + (1 - momentum) * xv
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # Implement the test-time forward pass for batch normalization.       #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = gamma * ((x - running_mean) / np.sqrt(running_var + eps)) + beta
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(cache['norm'] * dout, axis=0)
    N = dout.shape[0]
    path1 = 1 / cache['std'] * dout * cache['gamma'] - np.sum(1 / (cache['std'] * N) * dout * cache['gamma'], axis=0)
    path2 = 1 / N * cache['m'] * cache['std'] ** -3 * np.sum((cache['x'] - cache['m']) * dout * cache['gamma'], axis=0)
    path3 = 1 / N * cache['x'] * cache['std'] ** -3 * np.sum((cache['x'] - cache['m']) * dout * cache['gamma'], axis=0)
    dx = path1 + path2 - path3
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(cache['norm'] * dout, axis=0)
    N = dout.shape[0]
    dy = dout * cache['gamma']
    dx = 1 / (N * cache['std']) * (N * dy - np.sum(dy, axis=0) - cache['norm'] * np.sum(dy * cache['norm'], axis=0))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # Implement training phase forward pass for inverted dropout.         #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.random(x.shape)
        mask[mask >= p] = 1
        mask[mask < p] = 0
        out = mask * x / (1-p)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # Implement the test phase forward pass for inverted dropout.         #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # Implement training phase backward pass for inverted dropout         #
        #######################################################################
        dx = dout * mask / (1-dropout_param['p'])
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # Implement the convolutional forward pass.                               #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((H_, W_, F, N))
    p = np.transpose(np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))), (2, 3, 1, 0))
    for i in range(H_):
        for j in range(W_):
            out[i][j] = (np.dot(np.reshape(np.transpose(w, (2, 3, 1, 0)), (-1, F)).T, np.reshape(p[i*stride:i*stride+HH, j*stride:j*stride+WW], (-1, N))).T + b).T
    out = np.transpose(out, (3, 2, 0, 1))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # Implement the convolutional backward pass.                              #
    ###########################################################################
    (x, w, b, conv_param) = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    (N, F, H_, W_) = dout.shape
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    p = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    tran_d = np.transpose(dout, (2, 3, 0, 1))
    tran_w = np.transpose(w, (2, 3, 1, 0))

    db = np.sum(np.sum(np.reshape(dout, (N, F, -1)), axis=2), axis=0)
    dw = np.zeros((F, C, HH, WW))
    dp = np.zeros((H+2*pad, W+2*pad, C, N))
    for i in range(H_):
        for j in range(W_):
            dw += np.reshape(np.dot(tran_d[i][j].T, np.reshape(p[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW], (N, -1))), (F, C, HH, WW))
            dp[i*stride:i*stride+HH, j*stride:j*stride+WW] += np.reshape(np.dot(tran_d[i][j], tran_w.reshape(-1, F).T).T, (HH, WW, C, N))
    dx = np.transpose(dp, (3, 2, 0, 1))[:, :, pad:pad+H, pad:pad+W]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # Implement the max pooling forward pass                                  #
    ###########################################################################
    (N, C, H, W) = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    H_ = 1 + (H - HH) // stride
    W_ = 1 + (W - WW) // stride
    out = np.zeros((H_, W_, N, C))
    for i in range(H_):
        for j in range(W_):
            out[i][j] = np.max(np.reshape(x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW], (N, C, -1)), axis=2)
    out = np.transpose(out, (2, 3, 0, 1))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # Implement the max pooling backward pass                                 #
    ###########################################################################
    (N, C, H_, W_) = dout.shape
    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros(x.shape)
    for i in range(H_):
        for j in range(W_):
            ind = np.argmax(np.reshape(x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW], (N, C, -1)), axis=2)
            for n in range(N):
                for m in range(C):
                    dx[n][m][i*stride+ind[n][m]//WW][j*stride+ind[n][m]%WW] += dout[n][m][i][j]

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # Implement the forward pass for spatial batch normalization.             #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    out, cache = batchnorm_forward(np.reshape(np.transpose(x, (0, 2, 3, 1)), (-1, C)), gamma, beta, bn_param)
    out = np.transpose(np.reshape(out, (N, H, W, C)), (0, 3, 1, 2))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # Implement the backward pass for spatial batch normalization.            #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    (N, C, H, W) = dout.shape
    dx, dgamma, dbeta = batchnorm_backward_alt(np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C)), cache)
    dx = np.transpose(np.reshape(dx, (N, H, W, C)), (0, 3, 1, 2))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
