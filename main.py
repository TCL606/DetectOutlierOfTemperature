import scipy.io as scio
import numpy as np
import copy
import argparse

class DiscriminativeModel():

    def __init__(self, path, sigma):
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
                    self.graph[i, j] = self.lat[i] - self.lat[j] 
        self.W = np.exp(-self.graph ** 2 / 2 / sigma ** 2)


    def fit_data(self, t, i, day):
        local_var = 0
        for j in range(len(self.idx)):
            local_var += (t[j] - t[i]) ** 2 * self.W[i, j]
        return local_var        

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data.mat')
    parser.add_argument('--delta_t', type=int, default=20)
    parser.add_argument('--front', type=int, default=28)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--use_time_info', action='store_true')
    parser.add_argument('--sigma', type=float, default=3.0)
    args = parser.parse_args()

    path = args.path
    delta_t = args.delta_t
    front = args.front
    alpha = args.alpha
    use_time_info = args.use_time_info
    sigma = args.sigma

    model = DiscriminativeModel(path, sigma)
    preds = np.zeros([150, 31])
    labels = np.arange(150).repeat(31).reshape(150, 31)
    t = copy.deepcopy(model.treal)
    for j in range(31):
        for i in range(150):
            t[i, j] += delta_t
            if j > 0 and t[i, j] - np.average(t[i, max(j - front, 0):j]) > delta_t * alpha:
                if use_time_info:
                    preds[i, j] = i
                else:
                    preds[i, j] = preds[i, j] = model.detect_outlier(t[:, j], j)
            else:
                preds[i, j] = model.detect_outlier(t[:, j], j)
            t[i, j] -= delta_t
        acc_j = np.sum(labels[:, j] == preds[:, j]) / 150
        print(f'{j}: acc = {acc_j}')
    acc = np.sum(labels == preds) / 4650
    print(f'total: acc = {acc}')