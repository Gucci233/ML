import numpy as np
from sklearn.model_selection import KFold
from itertools import combinations_with_replacement
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
        
def polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    X=np.array(X)
    combinations=[]
    for d in range(1,degree+1):
        combinations+=[comb for comb in combinations_with_replacement(range(n_features), d)]
    
    n_output_features = len(combinations)+1
    # print(combinations,n_output_features)
    PF_array = np.empty((n_samples, n_output_features))
    PF_array[:, 0] = 1
    for i, comb in enumerate(combinations):
        PF_array[:, i+1] = X[:, comb].prod(1)

    return PF_array

class PolynomialRegression:
    def __init__(self, learning_rate=0.01, n_iterations=500, n_folds=5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.weights = None
        self.bias = None

        
    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=2)
        #print(type(X_poly))
        n_samples, n_features = X_poly.shape
        #print(X.shape,X_poly.shape)
        self.weights =np.ones(n_features).reshape(-1,1)
        self.bias = 0
        train_loss_list=[]
        val_loss_list=[]
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        #交叉验证
        for i in range(self.n_iterations):
            for train_idx, val_idx in kf.split(X_poly):
                X_train, y_train = X_poly[train_idx], y[train_idx]
                X_val, y_val = X_poly[val_idx], y[val_idx]
                X_train=np.array(X_train)
                y_train=np.array(y_train).reshape(-1,1)
                X_val=np.array(X_val)
                y_val=np.array(y_val).reshape(-1,1)
                y_pred_train = np.dot(X_train, self.weights) + self.bias
                y_pred_val = np.dot(X_val, self.weights) + self.bias
                train_loss = np.mean(np.square(y_pred_train- y_train))
                val_loss = np.mean(np.square(y_pred_val - y_val))
                dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_train - y_train))+ 0.005* np.sign(self.weights)#线性回归weight梯度
                db = (2 / n_samples) * np.sum(y_pred_train - y_train)#bias梯度
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        #draw("Poly loss",train_loss_list,val_loss_list)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=2)
        y_pred = np.dot(X_poly, self.weights) + self.bias
        return y_pred