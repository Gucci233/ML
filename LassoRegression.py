import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def draw(title,train_loss_list,val_loss_list):
        plt.figure()
        x = range(1, len(train_loss_list) + 1)
        plt.plot(x, train_loss_list, label = "train set loss")
        plt.plot(x, val_loss_list, label = "validation set loss")
        plt.legend()
        plt.xlabel("training times")
        plt.ylabel("loss value")
        plt.title(title)
        plt.show()


class LassoRegression:
    def __init__(self, learning_rate=0.01, n_iterations=500, n_folds=5,Lambda=0,iteration_method='GDT'):
        #Lambda是L1正则系数。默认为0，也就退化成线性回归,lasso回归就是比线性回归多了L1正则
        #iteration_method是使损失函数收敛的方法，GDT是梯度下降，CDT是坐标下降
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.Lambda=Lambda
        self.iteration_method=iteration_method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights =np.ones(n_features).reshape(-1,1)
        self.bias = 0
        train_loss_list=[]
        val_loss_list=[]
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        #交叉验证
        for n in range(self.n_iterations):
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                X_train=X_train.values
                y_train=np.array(y_train).reshape(-1,1)
                X_val=np.array(X_val)
                y_val=np.array(y_val).reshape(-1,1)
                y_pred_train = np.dot(X_train, self.weights) + self.bias
                y_pred_val = np.dot(X_val, self.weights) + self.bias
                train_loss = np.mean(np.square(y_pred_train- y_train))
                val_loss = np.mean(np.square(y_pred_val - y_val))
                if self.iteration_method=='CDT':
                    for i,item in enumerate(self.weights):
                        #在每一个W值上找到使损失函数收敛的点
                        #CDT可能会迭代速度很慢，因此这里只迭代50次，调大可能更接近于收敛，但是慢
                            for j in range(50):
                                y_pred_train = np.dot(X_train, self.weights) + self.bias
                                y_pred_val = np.dot(X_val, self.weights) + self.bias
                                dw = np.dot(X_train[:,i].T ,(y_pred_train - y_train))*(2 / n_samples)+ self.Lambda * np.sign(self.weights[i])
                                self.weights[i] -= dw* self.learning_rate
                                if abs(dw)<1e-3:
                                    break
                else:
                    dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_train - y_train))+self.Lambda* np.sign(self.weights)
                    self.weights -= self.learning_rate * dw
                db = (2 / n_samples) * np.sum(y_pred_train - y_train)
                self.bias -= self.learning_rate * db
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            if n % 100 == 0:
                print(f"Iteration {n}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        draw("LassoRegerssion loss",train_loss_list,val_loss_list)

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred