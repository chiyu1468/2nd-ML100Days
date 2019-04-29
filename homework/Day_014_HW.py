#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## 作業
# ### 請使用 application_train.csv, 根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram
#
# # [作業目標]
# - 試著調整資料, 並利用提供的程式繪製分布圖
#
# # [作業重點]
# - 如何將列出相異的 HOUSETYPE_MODE 類別 (In[3])
# - 如何依照不同的 HOUSETYPE_MODE 類別指定資料, 並繪製長條圖(.hist())? (In[3])

#%%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 忽略警告訊息
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = './data/'

#%%
# 讀取檔案
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
# app_train.head()

#%%
# 使用不同的 HOUSETYPE_MODE 類別繪製圖形, 並使用 subplot 排版
unique_house_type = app_train['HOUSETYPE_MODE'].unique()

x = len(unique_house_type)
nrows = 2
ncols = x // 2

plt.figure(figsize=(20,10))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)
    app_train.loc[app_train['HOUSETYPE_MODE']==str(unique_house_type[i]),'AMT_CREDIT'].hist()
    plt.title(str(unique_house_type[i]))
plt.show()


