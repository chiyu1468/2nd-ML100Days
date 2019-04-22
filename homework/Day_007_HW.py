#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 處理 outliers
# * 新增欄位註記
# * outliers 或 NA 填補
#     1. 平均數 (mean)
#     2. 中位數 (median, or Q50)
#     3. 最大/最小值 (max/min, Q100, Q0)
#     4. 分位數 (quantile)

#%%
# Import 需要的套件
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

# 設定 data_path
dir_data = './data/'

#%%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

#%% [markdown]
# ## 1. 列出 AMT_ANNUITY 的 q0 - q100
# ## 2.1 將 AMT_ANNUITY 中的 NAs 暫時以中位數填補
# ## 2.2 將 AMT_ANNUITY 的數值標準化至 -1 ~ 1 間
# ## 3. 將 AMT_GOOD_PRICE 的 NAs 以眾數填補
# 

#%%
# 1: 計算 AMT_ANNUITY 的 q0 - q100
columnName = 'AMT_ANNUITY'
print("呼叫 numpy 的方法:")
start_time = time.time()
q_all = [np.percentile(app_train[~app_train[columnName].isnull()]['AMT_ANNUITY'], q = i) for i in range(101)]
print("Elapsed time: %.3f secs" % (time.time() - start_time))
pd.DataFrame({'q': list(range(101)),
              'value': q_all})

#%%
columnName = 'AMT_ANNUITY'
print("Pandas 內建的屬性:")
start_time = time.time()
q_all = [app_train['AMT_ANNUITY'].quantile(i) for i in np.arange(0,1.01,0.01)]
print("Elapsed time: %.3f secs" % (time.time() - start_time))

#%%
# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

q_50 = np.percentile(app_train[~app_train[columnName].isnull()]['AMT_ANNUITY'], q = 50)
app_train.loc[app_train[columnName].isnull(),columnName] = q_50

print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

#%% [markdown]
# ### Hints: Normalize function (to -1 ~ 1)
# $ y = 2*(\frac{x - min(x)}{max(x) - min(x)} - 0.5) $

#%%
# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

def normalize_value(x):
    """
    Your Code Here, compelete this function
    """
    min = x.min()
    max = x.max()
    return x.apply(lambda v : ((v - min)/(max - min) - 0.5)*2)

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
app_train['AMT_ANNUITY_NORMALIZED'].describe()

#%%
# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
from collections import defaultdict
mode_dict = defaultdict(lambda:0)
columnName = 'AMT_GOODS_PRICE'

for value in app_train[~app_train[columnName].isnull()][columnName]:
    mode_dict[value] += 1
    
mode_get = sorted(mode_dict.items(), key=lambda kv: kv[1], reverse=True)

value_most = mode_get[0]
print("The value repeat most is : {} , it repeat {} times.".format(value_most[0],value_most[1]))

# mode_goods_price = list(app_train[columnName].value_counts().index)
app_train.loc[app_train[columnName].isnull(), columnName] = value_most[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

