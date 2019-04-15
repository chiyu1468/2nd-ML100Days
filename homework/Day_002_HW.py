#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	# print(os.getcwd())
except:
	pass
#%% [markdown]
# ## 練習時間
# 資料的操作有很多，接下來的馬拉松中我們會介紹常被使用到的操作，參加者不妨先自行想像一下，第一次看到資料，我們一般會想知道什麼訊息？
# 
# #### Ex: 如何知道資料的 row 數以及 column 數、有什麼欄位、多少欄位、如何截取部分的資料等等
# 
# 有了對資料的好奇之後，我們又怎麼通過程式碼來達成我們的目的呢？
# 
# #### 可參考該[基礎教材](https://bookdata.readthedocs.io/en/latest/base/01_pandas.html#DataFrame-%E5%85%A5%E9%97%A8)或自行 google

#%%
import os
import numpy as np
import pandas as pd

#%%
# 設定 data_path
dir_data = './data/'

#%%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

#%% [markdown]
# ### 如果沒有想法，可以先嘗試找出剛剛例子中提到的問題的答案
# #### 資料的 row 數以及 column 數

#%%
row, column = app_train.shape
print("row num : {} \ncolumn num : {}".format(row,column))

#%% [markdown]
# #### 列出所有欄位

#%%
print("Below are all columns item :")
column_items = app_train.axes[1]
strLimit = 100
str = ""
for item in column_items:
    if len(str.expandtabs())+len(item) > strLimit:
        print(str)
        str = ""
    if str != "":
        str = str + ',\t' + item
    str += item
print(str)

#%% [markdown]
# #### 條件式資料截取

#%%
condSeries = (app_train["SK_ID_CURR"] < 100017)
app_train.loc[condSeries]

#%% [markdown]
# #### 還有各種數之不盡的資料操作，重點還是取決於實務中遇到的狀況和你想問的問題，在馬拉松中我們也會陸續提到更多例子
# #### 相關操作可以看這裡
# #### https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html



