#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass

#%%
# Import 需要的套件
import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = './data/'


#%%
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)


#%%
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#%% [markdown]
# ## 練習時間
#%% [markdown]
# 觀察有興趣的欄位的資料分佈，並嘗試找出有趣的訊息
# #### Eg
# - 計算任意欄位的平均數及標準差
# - 畫出任意欄位的[直方圖](https://zh.wikipedia.org/zh-tw/%E7%9B%B4%E6%96%B9%E5%9B%BE)
# 
# ### Hints:
# - [Descriptive Statistics For pandas Dataframe](https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)
# - [pandas 中的繪圖函數](https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html)
# 

#%%
item = 'DAYS_REGISTRATION'
mask = ~app_train[item].isnull()
sub_train = app_train.loc[mask,item]
sub_train = np.abs(sub_train)
sub_train.describe()

#%%
div_num = 40
# day_dist = [0 for i in range(div_num)]
# interval = np.ceil((sub_train.max() - sub_train.min()) / div_num)
# sub_train.apply(lambda x : np.floor_divide(x,interval))
sub_train.hist(bins = div_num)