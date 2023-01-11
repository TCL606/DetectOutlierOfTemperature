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
        for i in range(len(self.idx)):
            for j in range(len(self.idx)):
                if i != j:
                    self.graph[i, j] = math.acos(
                        math.cos(self.lng[i] - self.lng[j]) *
                        math.cos(self.lat[i]) * math.cos(self.lat[j]) +
                        math.sin(self.lat[i]) * math.sin(self.lat[j])
                    )

    def fit_data(self, t, i):
        t_temp = np.concatenate([t[:i], t[i + 1:]])
        # lat_temp = np.concatenate([self.lat[:i], self.lat[i + 1:]])
        # lng_temp = np.concatenate([self.lng[:i], self.lng[i + 1:]])
        # lr = self.fit_data(t_temp, lat_temp, lng_temp)
        # t_pred = lr.predict(np.array([self.lat[i], self.lng[i]]).reshape(1, -1))

        dist = np.concatenate([self.graph[i, :i], self.graph[i, i + 1:]])
        dist_down = 1 / dist
        prob = dist_down / np.sum(dist_down)
        # prob = torch.softmax(torch.tensor(dist_down), dim=0).numpy()
        t_pred = np.sum(t_temp * prob)

        return t_pred


    def detect_outlier(self, t, lat=None, lng=None):
        if lat is None:
            lat = self.lat
        if lng is None:
            lng = self.lng

        assert t.shape == lat.shape
        assert t.shape == lng.shape

        delta = np.zeros(t.shape)
        for i in range(lat.shape[0]):
            t_pred = self.fit_data(t, i)
            delta[i] = t_pred - t[i]

        pred = np.argmax(np.abs(delta))

        return pred


if __name__ == '__main__':
    path = '../data.mat'
    model = DiscrimativeModel(path)
    preds = np.zeros([150, 31])
    t = copy.deepcopy(model.treal)
    for i in range(150):
        for j in range(31):
            t[i, j] += 20
            preds[i, j] = model.detect_outlier(t[:, j])
            t[i, j] -= 20
    labels = np.arange(150).repeat(31).reshape(150, 31)
    acc = np.sum(labels == preds) / 4650
    print(acc)
