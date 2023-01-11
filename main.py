import scipy.io as scio
import numpy as np
import copy
import argparse

class DiscriminativeModel():

    def __init__(self, path, sigma):
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

    def analyze_space(self, t, day, lat=None, lng=None):
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
    
    def analyze_time(self, t, day, threhold):
        delta = t[:, day] - np.average(t[:, max(day - front, 0): day])
        pred = np.argmax(np.abs(delta))
        if abs(delta[pred]) >= threhold:
            return pred
        else: 
            return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data.mat')
    parser.add_argument('--delta_t', type=int, default=20)
    parser.add_argument('--front', type=int, default=10)
    parser.add_argument('--threhold', type=float, default=50)
    parser.add_argument('--use_time_info', action='store_true')
    parser.add_argument('--sigma', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--rand_add', action='store_true', help="whether to randomly changing temperature")
    args = parser.parse_args()

    path = args.path
    delta_t = args.delta_t
    front = args.front
    threhold = args.threhold
    use_time_info = args.use_time_info
    sigma = args.sigma
    seed = args.seed
    random_add = args.rand_add

    np.random.seed(seed)

    model = DiscriminativeModel(path, sigma)
    preds = np.zeros([150, 31])
    labels = np.arange(150).repeat(31).reshape(150, 31)
    t = copy.deepcopy(model.treal)
    for j in range(31):
        for i in range(150):
            if random_add:
                pm = 1 if np.random.randint(0, 2) == 0 else -1
            else:
                pm = 1
            t[i, j] += delta_t * pm
            if j >= front:
                if use_time_info:
                    time_predict = model.analyze_time(t, j, threhold)
                    if time_predict == -1:
                        preds[i, j] = model.analyze_space(t[:, j], j)
                    else:
                        preds[i, j] = time_predict
                else:
                    preds[i, j] = model.analyze_space(t[:, j], j)
            else:
                preds[i, j] = model.analyze_space(t[:, j], j)
            t[i, j] -= delta_t * pm
        acc_j = np.sum(labels[:, j] == preds[:, j]) / 150
        print(f'{j}: acc = {acc_j}')
    acc = np.sum(labels == preds) / 4650
    print(f'total: acc = {acc}')