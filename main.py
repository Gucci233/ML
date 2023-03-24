import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from itertools import combinations_with_replacement
from KNN import KnnRegression
from PolynomialRegression import PolynomialRegression
from MultivariateLinearRegression import MultivariateLinearRegression
from LassoRegression import LassoRegression


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

#plt.figure()
x_list = []
before_list = []
for i in range(1, len(train['Item_Weight'] + 1)):
    if train['Item_Weight'].notna()[i]:
        x_list.append(i)
        before_list.append(train['Item_Weight'][i])
# plt.scatter(x_list, before_list, s = 0.5, label = 'Item_Weight before filled')
# plt.legend()
# plt.xlabel("train set number")
# plt.ylabel("weight")
# plt.title("Item_Weight before and after filled")


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

# x_list0 = []
# after_list = []
# for i in range(1, len(train['Item_Weight'] + 1)):
#     if i not in x_list:
#         x_list0.append(i)
#         after_list.append(train['Item_Weight'][i])
# plt.scatter(x_list0, after_list, s = 0.5, label = 'Item_Weight after filled')
# plt.legend()
# plt.show()

#用最大最小归一化的方法来归一化训练集
for f in x_train:
    if np.mean(x_train[f])>1.0:
        x_train[f]=(x_train[f]-np.min(x_train[f]))/(np.max(x_train[f])-np.min(x_train[f]))
        x_test[f]=(x_test[f]-np.min(x_test[f]))/(np.max(x_test[f])-np.min(x_test[f]))
        

l=LinearRegression()
l.fit(x_train,y_train)
l.predict(x_test)
y_pred_linear=l.predict(x_test)


y_train=y_train/1000#防止溢出，进行放缩处理
print("下面开始多元线性回归训练")
model3=MultivariateLinearRegression(n_iterations=1000)
model3.fit(x_train,y_train)

print("下面开始knn回归训练")
model2 = KnnRegression(k=5)
y_test_knn = model2.predict(x_train, y_train, x_test[0:100])
print("下面开始多项式回归训练")
model= PolynomialRegression()
model.fit(x_train,y_train)

print("下面开始lasso回归训练")
model1=LassoRegression(Lambda=0.01,iteration_method='CDT')
model1.fit(x_train,y_train)

y_test_linear_diy=model3.predict(x_test[0:100])
y_test_lasso=model1.predict(x_test[0:100])
y_test_poly=model.predict(x_test[0:100])

print('调库线性回归预测结果')
print(y_pred_linear)
print('手写线性回归模型训练结果')
print(y_test_linear_diy*1000)
print('knn模型训练结果')
print(y_test_knn*1000)
print('手写多项式模型预测结果：')
print(y_test_poly*1000)
print('手写lasso回归模型预测结果：')
print(y_test_lasso*1000)

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.xticks(np.linspace(0,10000,11))
plt.yticks(np.linspace(0,6000,11))
x=range(1,len(y_test_poly)+1)

plt.plot(x,y_pred_linear[0:100],color='r',label="库函数线性回归")
plt.plot(x,y_test_poly*1000,color='g',label="手写多项式回归")
plt.plot(x,y_test_lasso*1000,color='b',label="手写lasso回归")
plt.plot(x,y_test_linear_diy*1000,color='m',label="手写线性回归")
plt.plot(x,y_test_knn*1000,color='y',label="手写knn回归")
plt.title("预测值曲线")
plt.legend(loc="upper right")
plt.show()

x_data=["linear_loss_mean","knn_loss_mean","poly_loss_mean","lasso_loss_mean"]
y_data=[np.mean((y_test_linear_diy-y_pred_linear[0:100]/1000)**2),np.mean((y_test_knn-y_pred_linear[0:100]/1000)**2),
        np.mean((y_test_poly-y_pred_linear[0:100]/1000)**2),np.mean((y_test_lasso-y_pred_linear[0:100]/1000)**2)]
plt.bar(x_data,y_data)
 
plt.title("各分类平均loss")
plt.xlabel("分类模型")

plt.show()




