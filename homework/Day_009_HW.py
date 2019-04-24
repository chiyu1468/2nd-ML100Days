
#%%  [markdown]
# 作業
# - 參考範例程式碼，模擬一組負相關的資料，並計算出相關係數以及畫出 scatter plot
# - 仿照 In[4], In[5] 的語法, 寫出負相關的變數, 並觀察相關矩陣以及分布圖

#%%
# 載入基礎套件
import numpy as np
np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#%% [markdown]
# ### 負相關

#%%
# 隨機生成兩組 1000 個介於 0~50 的數的整數 x, y, 看看相關矩陣如何
x = np.random.randint(0, 50, 1000)
y = 25 - x + np.random.normal(0, 15, 1000)

#%%
# 呼叫 numpy 裡的相關矩陣函數 (corrcoef)
np.corrcoef(x, y)

#%%
# 將分布畫出來看看吧
plt.scatter(x, y)


