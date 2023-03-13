import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from itertools import combinations_with_replacement
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    X=np.array(X)
    combinations=[]
    for d in range(1,degree+1):
        combinations+=[comb for comb in combinations_with_replacement(range(n_features), d)]
    
    n_output_features = len(combinations)+1
    print(combinations,n_output_features)
    PF_array = np.empty((n_samples, n_output_features))
    PF_array[:, 0] = 1
    for i, comb in enumerate(combinations):
        PF_array[:, i+1] = X[:, comb].prod(1)

    return PF_array

class PolynomialRegression:
    def __init__(self, learning_rate=0.01, n_iterations=2000, n_folds=5):
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
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        X_poly = polynomial_features(X, degree=2)
        y_pred = np.dot(X_poly, self.weights) + self.bias
        return y_pred

class MultivariateLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=2000, n_folds=5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights =np.ones(n_features).reshape(-1,1)
        self.bias = 0
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        #交叉验证
        for i in range(self.n_iterations):
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
                dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_train - y_train))#线性回归weight梯度
                db = (2 / n_samples) * np.sum(y_pred_train - y_train)#bias梯度
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


class LassoRegression:
    def __init__(self, learning_rate=0.01, n_iterations=2000, n_folds=5,Lambda=0,iteration_method='GDT'):
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
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        #交叉验证
        for i in range(self.n_iterations):
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
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


#以下为特征工程
train = pd.read_csv('sale prediction/train.csv')
test = pd.read_csv('sale prediction/test.csv')
submission=pd.read_csv('sale prediction/sample_submission.csv')
features=['row_id','cfips','county','state','first_day_of_month','active']

df = pd.concat([train, test])#合并测试集和训练集，做同样处理

df['Outlet_Size_Medium']=np.zeros(df.shape[0])
df['Outlet_Size_High']=np.zeros(df.shape[0])
df['Outlet_Size_Small']=np.zeros(df.shape[0])
df['Outlet_Size_Non']=np.zeros(df.shape[0])
df.Outlet_Size_Medium[df.Outlet_Size=='Medium']=1
df.Outlet_Size_High[df.Outlet_Size=='High']=1
df.Outlet_Size_Small[df.Outlet_Size=='Small']=1
df.Outlet_Size_Non[df.Outlet_Size=='Non']=1

df['year_sin']=np.sin(df['Outlet_Establishment_Year'])
df['year_cos']=np.cos(df['Outlet_Establishment_Year'])

encode_feats=['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type']
label='Item_Outlet_Sales'

df = df.sort_values(by=['Outlet_Location_Type', 'Outlet_Establishment_Year']).reset_index(drop=True)
df['ts'] = df['Outlet_Establishment_Year'].values.astype(np.int64) // 100
df['ts1'] = df.groupby('Outlet_Location_Type')['ts'].shift(1)
df['ts_diff1'] = df['ts1'] - df['ts']

for f in encode_feats:
    le = LabelEncoder()
    df[f] = le.fit_transform(df[f])
    df[f+'_ts_diff_mean'] = df.groupby([f])['ts_diff1'].transform('mean')
    df[f+'_ts_diff_std'] = df.groupby([f])['ts_diff1'].transform('std')


train = df[df[label].notna()].reset_index(drop=True)
test = df[df[label].isna()].reset_index(drop=True)
feats = [f for f in test if f not in [label, 'Item_Identifier', 'Outlet_Identifier', 'Outlet_Size', 'ts', 'ts1','ts_diff1']]
x_train = train[feats]
y_train = train[label]
x_test=test[feats]

#因为Item_Weight中有空值，这里通过MLP神经网络来预测空值
XX_train=train[train['Item_Weight'].notna()].reset_index(drop=True)
YY_train=XX_train['Item_Weight']
XX_val=train[train['Item_Weight'].isna()].reset_index(drop=True)
new_f=['Item_Fat_Content','Item_Visibility','Item_MRP','Item_Type','Outlet_Size_Medium','Outlet_Size_High','Outlet_Size_Small','Outlet_Size_Non','Item_Outlet_Sales']
new_XX_train=XX_train[new_f]
new_XX_val=XX_val[new_f]
mlp = MLPRegressor()
mlp.fit(new_XX_train,YY_train)
YY_val=mlp.predict(new_XX_val)
XX_val['Item_Weight']=YY_val
train = pd.concat([XX_train, XX_val]).reset_index(drop=True)
x_train=train[feats]
y_train=train[label]
x_test=x_test[x_test['Item_Weight'].notna()]
x_train.to_csv('cat.csv', index=False)

#用最大最小归一化的方法来归一化训练集
for f in x_train:
    if np.mean(x_train[f])>1.0:
        x_train[f]=(x_train[f]-np.min(x_train[f]))/(np.max(x_train[f])-np.min(x_train[f]))
        x_test[f]=(x_test[f]-np.min(x_test[f]))/(np.max(x_test[f])-np.min(x_test[f]))
        
# model= MultivariateLinearRegression()
# model.fit(x_train,y_train)
model1=LassoRegression(Lambda=36,iteration_method='CDT')
model1.fit(x_train,y_train)
l=LinearRegression()
l.fit(x_train,y_train)
l.predict(x_test)
y_pred_linear=l.predict(x_test)


y_train=y_train/1000#防止溢出，进行放缩处理
model= PolynomialRegression()
model.fit(x_train,y_train)

model1=LassoRegression(Lambda=0.1,iteration_method='CDT')
model1.fit(x_train,y_train)

y_test_lasso=model1.predict(x_test)
y_test=model.predict(x_test)

print('调库线性回归预测结果')
print(y_pred_linear)
print('手写多项式模型预测结果：')
print(y_test*1000)
print('手写lasso回归模型预测结果：')
print(y_test_lasso*1000)



