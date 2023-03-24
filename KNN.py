import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class KnnRegression:
    def __init__(self, k=1, n_iterations=10, n_folds=5):
        self.k = k
        self.n_folds=n_folds
        self.n_iterations = n_iterations

        
    def Calculate(self, test, X_train, y_train):
        m, n = X_train.shape
        # 计算测试数据到每个点的欧式距离
        distances = []
        for i in range(m):
            sum = 0
            for j in range(n):
                sum += (test[j] - X_train[i][j]) ** 2
            distances.append(sum ** 0.5)
        list0 = range(m)
        distances, list0 = zip(*sorted(zip(distances, list0)))
        sum = 0  
        for i in range(self.k):
            sum += y_train[list0[i]]
        return sum/self.k

    def fit(self, X, y):
        n_samples, n_features = X.shape
        kf = KFold(n_splits=5, shuffle=True)
        #交叉验证, 寻找最佳K值
        klist = []
        for i in range(self.n_iterations ):
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                X_train=X_train.values
                y_train=np.array(y_train).reshape(-1,1)
                X_val=np.array(X_val)
                y_val=np.array(y_val).reshape(-1,1)
                
                #对每一个k值求出验证集的总误差
                m, n = X_val.shape
                loss = 0
                for j in range(m):
                    predict = self.Calculate(X_val[j], X_train, y_train, i)
                    loss = loss + (predict - y_val[j]) ** 2
            klist.append(loss)
        sklist = sorted(klist)
        self.k = klist.index(sklist[0]) + 1

    def predict(self, X, y, test):
        ans = np.array([])
        X_train = X.values
        y_train = np.array(y).reshape(-1, 1)
        test0 = np.array(test)

        for i in test0:
            ans = np.append(ans, self.Calculate(i, X_train, y_train))
        return ans