import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,LassoCV
import lightgbm as lgb

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

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
        n=0
        kf = KFold(n_splits=self.n_folds, shuffle=True)
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
                y_pred_train_logic=sigmoid(y_pred_train)
                y_pred_val_logic=sigmoid(y_pred_val)
                train_loss = np.mean(np.square(y_pred_train- y_train))
                val_loss = np.mean(np.square(y_pred_val - y_val))
                dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_train - y_train))#线性回归梯度
                db = (2 / n_samples) * np.sum(y_pred_train - y_train)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


class LassoRegression:
    def __init__(self, learning_rate=0.01, n_iterations=2000, n_folds=5,Lambda=0,iteration_method='GDT'):
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
        n=0
        kf = KFold(n_splits=self.n_folds, shuffle=True)
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
                            for j in range(50):
                                y_pred_train = np.dot(X_train, self.weights) + self.bias
                                y_pred_val = np.dot(X_val, self.weights) + self.bias
                                dw = np.dot(X_train[:,i].T ,(y_pred_train - y_train))*(2 / n_samples)+ self.Lambda * np.sign(self.weights[i])
                                self.weights[i] -= dw* self.learning_rate
                                if abs(dw)<1e-3:
                                    break
                else:
                    dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_train - y_train))+self.Lambda* np.sign(self.weights)#线性回归梯度
                    self.weights -= self.learning_rate * dw
                db = (2 / n_samples) * np.sum(y_pred_train - y_train)
                self.bias -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Iteration {i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred



train = pd.read_csv('sale prediction/train.csv')
test = pd.read_csv('sale prediction/test.csv')
submission=pd.read_csv('sale prediction/sample_submission.csv')
features=['row_id','cfips','county','state','first_day_of_month','active']

df = pd.concat([train, test])

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


model= MultivariateLinearRegression()
model1=LassoRegression(Lambda=36,iteration_method='CDT')
#y_train=np.log10(y_train)#防止溢出，进行放缩处理
#归一化
for f in x_train:
    if np.mean(x_train[f])>1.0:
        x_train[f]=(x_train[f]-np.min(x_train[f]))/(np.max(x_train[f])-np.min(x_train[f]))
        x_test[f]=(x_test[f]-np.min(x_test[f]))/(np.max(x_test[f])-np.min(x_test[f]))
model1.fit(x_train,y_train)
y_test=model1.predict(x_test)
print('手写模型预测结果：')
print(y_test)

l=LinearRegression()
l.fit(x_train,y_train)
l.predict(x_test)
y_pred_linear=l.predict(x_test)
print('调库线性回归预测结果')
print(y_pred_linear)


