import numpy as np


def convolve_3x3(array, weights):
    """ Convolve array with weights.

    Args:
        array: Array to be convolved, [batch_size, width, height, in_depth]
        weights: Convolution filter,
                 [filter_wd, filter_ht, in_depth, out_depth]
    Returns:
        Convolved array, [batch_size, width, height, out_depth]
    """
    patch = get_3x3_patches(array)
    # Faster method to contract ∑_a ∑_b ∑_c patches_{nwhabc} weight_{abcd}:
    return np.dot(patch.reshape(*patch.shape[:3], -1),
                  weights.reshape(-1, weights.shape[-1]))


def get_3x3_patches(array):
    """ Get 3 by 3 patches of an array (with padding).

    Args:
        array: [batch_size, width, height, depth]
    Returns:
        An array of shape [batch_size, width, height, 3, 3, depth]
        where the dimensions with 3s are the 9 different patches.
    """
    batch_size, width, height, depth = array.shape

    # pad height and width with zeros.
    pad = ((0, 0), (1, 1), (1, 1), (0, 0))
    padded = np.pad(array, pad, 'constant', constant_values=0)
    s_batch, s_width, s_height, s_depth = padded.strides

    shape = (batch_size, width, height, 3, 3, depth)
    strides = (s_batch, s_width, s_height, s_width, s_height, s_depth)

    return np.lib.stride_tricks.as_strided(
        padded, writeable=False, shape=shape, strides=strides)


def _subarr_slice(i):
    """ For indices i={0...3} return the corresponding sub-array slice."""
    return np.index_exp[:, (i>>1)::2, i&1::2, :]

def max_pool(array):
    """ Max pool an array to half the size."""
    # Contorted to fit into speedy list comprehension.
    return np.amax([array[_subarr_slice(i)] for i in range(4)], axis=0)

def max_pool_indices(array):
    """ Get indices of pooled array values."""
    return np.argmax([array[_subarr_slice(i)] for i in range(4)], axis=0)

def distribute_to_indices(indices, distrib, shape):
    """ Return an array of shape with indices filled in by distrib."""
    ret = np.empty(shape, dtype=distrib.dtype)
    for i in range(4):
        ret[_subarr_slice(i)] = np.where(indices == i, distrib, 0)
    return ret
