#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # [EDA] 了解變數分布狀態: Bar & KDE (density plot)
#%% [markdown]
# # To do: 變項的分群比較
# 1. 自 20 到 70 歲，切 11 個點，進行分群比較 (KDE plot)
# 2. 以年齡區間為 x, target 為 y 繪製 barplot
#
# # [作業目標]
# - 試著調整資料, 並利用提供的程式繪製分布圖
#
# # [作業重點]
# - 如何將資料依照歲數, 將 20 到 70 歲切成11個區間? 
# (In[4], Hint : 使用 numpy.linspace),  
#   送入繪圖前的除了排序外, 還要注意什麼? (In[5])
# - 如何調整對應資料, 以繪製長條圖(bar chart)? (In[7])
# # [參考資料]
# 1. [Intro to Kernel Density Estimation](https://www.youtube.com/watch?v=x5zLaWT5KPs) 
# 2. [Density Estimationand Mixture Models](https://nic.schraudolph.org/teach/ml03/ML_Class4.pdf)

#%%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

# 忽略警告訊息
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = './data/'

#%%
# 讀取檔案
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

#%%
# 資料整理 ( 'DAYS_BIRTH'全部取絕對值 )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

#%%
# 根據年齡分成不同組別 (年齡區間 - 還款與否)
age_data = app_train[['TARGET', 'DAYS_BIRTH']] # subset
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365 # day-age to year-age

#自 20 到 70 歲，切 11 個點 (得到 10 組)
bin_cut = np.linspace(20,70,11)
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut) 

# 顯示不同組的數量
print(age_data['YEARS_BINNED'].value_counts())
age_data.head()

#%%
# 繪圖前先排序 / 分組
# 懶得想變數名 所以乾脆寫成一行
# 一張圖很難看，拆成十張圖 
year_group_sorted = [_ for _ in age_data.groupby('YEARS_BINNED').grouper.result_index]

for i in range(len(year_group_sorted)):
    plt.figure(i,figsize=(8,6))
    plt.clf()
    plt.title('KDE with Age groups : {}'.format(year_group_sorted[i]))
    # 不是討債對象
    sns.distplot(age_data.loc[
        (age_data['YEARS_BINNED'] == year_group_sorted[i]) &
        (age_data['TARGET'] == 0), 'YEARS_BIRTH'
        ], 
        label = "NOT TARGET")
    # 是討債對象
    sns.distplot(age_data.loc[
        (age_data['YEARS_BINNED'] == year_group_sorted[i]) &
        (age_data['TARGET'] == 1), 'YEARS_BIRTH'
        ], 
        label = "TARGET")
    plt.legend()
plt.show()

#%%
# 計算每個年齡區間的 Target、DAYS_BIRTH與 YEARS_BIRTH 的平均值
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups

#%%
plt.figure(figsize = (8, 8))

# 以年齡區間為 x, target 為 y 繪製 barplot
# https://seaborn.pydata.org/generated/seaborn.barplot.html
# f_data = age_data.loc[age_data['TARGET'] == 1]
px = 'YEARS_BINNED'
py = 'TARGET'
sns.barplot(px, py, data=age_data)

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


