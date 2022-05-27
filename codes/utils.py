from typing import Tuple

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def topk_element_in_ndarray(matrix: np.ndarray, topk: int):
    """ 在矩阵中寻找topk元素（无序），同时返回坐标 """
    coo: Tuple[np.ndarray] = np.unravel_index(np.argpartition(matrix.ravel(), -topk)[-topk:], matrix.shape)
    val: np.ndarray = matrix[coo]
    return val, coo
