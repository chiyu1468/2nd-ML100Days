#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
import warnings
warnings.filterwarnings('ignore')
#%% [markdown]
# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察均值編碼的效果
#%% [markdown]
# # [作業重點]
# - 仿造範例, 完成標籤編碼與均值編碼搭配邏輯斯迴歸的預測
# - 觀察標籤編碼與均值編碼在特徵數量 / 邏輯斯迴歸分數 / 邏輯斯迴歸時間上, 分別有什麼影響 (In[3], Out[3], In[4], Out[4]) 
#%% [markdown]
# # 作業1
# * 請仿照範例，將鐵達尼範例中的類別型特徵改用均值編碼實作一次

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data_path = 'data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
# df.head()
df.shape

#%%
#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
# df.head()
df.shape

#%% [markdown]
# # 作業2
# * 觀察鐵達尼生存預測中，均值編碼與標籤編碼兩者比較，哪一個效果比較好? 可能的原因是什麼?
# > 這個例子根本已經 overfit 怎麼比較啦。
# > Survived 是一個 True / False 的問題一維的二元變量
# > 還用 Mean Encoding 去降低 Logistic Regression 的難度..........


#%%
# 對照組 : 標籤編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')

#%%
# 均值編碼 + 邏輯斯迴歸
data = pd.concat([df[:train_num], train_Y], axis=1)
for c in df.columns:
    mean_df = data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    data = pd.merge(data, mean_df, on=c, how='left')
    data = data.drop([c] , axis=1)
data = data.drop(['Survived'] , axis=1)
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, data, train_Y, cv=5)}')
print(f'time : {time.time() - start} sec')

