#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 作業
# - 新增一個欄位 `customized_age_grp`，把 `age` 分為 (0, 10], (10, 20], (20, 30], (30, 50], (50, 100] 這五組，
# '(' 表示不包含, ']' 表示包含  
# - Hints: 執行 ??pd.cut()，了解提供其中 bins 這個參數的使用方式
# - 跟 Day11 用一樣的語法? 是我誤解了啥莫???
#
# # [作業目標]
# - 請同學試著查詢 pandas.cut 這個函數還有哪些參數, 藉由改動參數以達成目標
# - 藉由查詢與改動參數的過程, 熟悉查詢函數的方法與理解參數性質, 並了解數值的離散化的調整工具
#
# # [作業重點]
# - 仿照 In[3], In[4] 的語法, 並設定 pd.cut 的參數以指定間距


#%%
# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

#%%
# 初始設定 Ages 的資料
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})


#%%
# 作業
bin_cut = np.array([0.,10., 20., 30., 50., 100.,])
ages['customized_age_grp'] = pd.cut(ages["age"], bins = bin_cut) 

#%%
ages.groupby('customized_age_grp').size()
