import numpy as np


def temporal_consistency(mask_stack, window=2):

    T,H,W = mask_stack.shape

    output = np.zeros((H,W))

    for t in range(T):

        start = max(0,t-window)
        end   = min(T,t+window+1)

        local = mask_stack[start:end]

        valid = local.sum(axis=0) >= 2

        output[valid] = 1

    return output