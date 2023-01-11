import scipy.io as scio
import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import math
import torch

class DiscrimativeModel():

    def __init__(self, path):
        path = '../data.mat'
        self.data = scio.loadmat(path)
        if type(self.data) is dict:
            self.data = self.data['data']

        self.idx = self.data[:, 0]
        self.treal = self.data[:, 1:32]
        self.lat = self.data[:, 32]
        self.lng = self.data[:, 33]

        self.graph = np.zeros([len(self.idx), len(self.idx)])
        self.W = np.zeros([len(self.idx), len(self.idx)])
        for i in range(len(self.idx)):
            for j in range(len(self.idx)):
                if i != j:
                    self.graph[i, j] = math.acos(
                        math.cos(self.lng[i] - self.lng[j]) *
                        math.cos(self.lat[i]) * math.cos(self.lat[j]) +
                        math.sin(self.lat[i]) * math.sin(self.lat[j])
                    )
        self.W = np.exp(-self.graph ** 2 / 2 / 5 ** 2)


    def fit_data(self, t, i, day):
        # t_temp = np.concatenate([t[:i], t[i + 1:]])
        # dist = torch.from_numpy(np.concatenate([self.graph[i, :i], self.graph[i, i + 1:]]))
        # k = 149
        # dist_mink, idx_mink = torch.topk(dist, k, largest=False)
        # dist_down = 1 / dist_mink.numpy()
        # prob = dist_down / np.sum(dist_down)
        # t_pred = np.sum(t_temp[idx_mink] * prob)
        # return t_pred - t[i]

        local_var_loss = 0
        for j in range(len(self.idx)):
            local_var_loss += (t[j] - t[i]) ** 2 * self.W[i, j]
        return local_var_loss        

    def detect_outlier(self, t, day, lat=None, lng=None):
        if lat is None:
            lat = self.lat
        if lng is None:
            lng = self.lng

        assert t.shape == lat.shape
        assert t.shape == lng.shape

        delta = np.zeros(t.shape)
        for i in range(lat.shape[0]):
            delta_pred = self.fit_data(t, i, day)
            delta[i] = delta_pred
        
        pred = np.argmax(np.abs(delta))

        return pred


if __name__ == '__main__':
    path = '../data.mat'
    model = DiscrimativeModel(path)
    preds = np.zeros([150, 31])
    labels = np.arange(150).repeat(31).reshape(150, 31)
    t = copy.deepcopy(model.treal)
    delta_t = 20
    front = 28
    alpha = 0.5
    for j in range(31):
        for i in range(150):
            t[i, j] += delta_t
            if j > 0 and t[i, j] - np.average(t[i, max(j - front, 0):j]) > delta_t * alpha:
                preds[i, j] = preds[i, j] = model.detect_outlier(t[:, j], j) # i
            else:
                preds[i, j] = model.detect_outlier(t[:, j], j)
            t[i, j] -= delta_t
        acc_j = np.sum(labels[:, j] == preds[:, j]) / 150
        print(f'{j}: acc = {acc_j}')
    acc = np.sum(labels == preds) / 4650
    print(f'total: acc = {acc}')