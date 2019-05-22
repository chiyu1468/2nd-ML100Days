
#%% [markdown]
# ## 練習時間
# 假設我們資料中類別的數量並不均衡，在評估準確率時可能會有所偏頗，試著切分出 y_test 中，0 類別與 1 類別的數量是一樣的 (亦即 y_test 的類別是均衡的)

#%%
import numpy as np
from sklearn.model_selection import train_test_split, KFold
X = np.arange(1000).reshape(200, 5)
y = np.zeros(200)
y[:40] = 1

#%% [markdown]
# 可以看見 y 類別中，有 160 個 類別 0，40 個 類別 1 ，請試著使用 train_test_split 函數，
# 切分出 y_test 中能各有 10 筆類別 0 與 10 筆類別 1 。(HINT: 參考函數中的 test_size，可針對不同類別各自作切分後再合併)
# > 我想題目應該改成按照比例切分會比較合乎邏輯
# > 不然會使 Training Data 有"濃度"差異，進而造成訓練出來的模型有未知的 variance
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)

#%%
y_test

#%%
# However, this is homework. so just do it~
X_1 = X[y == 1]
X_0 = X[y == 0]
y_1 = y[y == 1]
y_0 = y[y == 0]

_, X0_test, _, y0_test = train_test_split(
    X_0, y_0, test_size=10, random_state=42)
_, X1_test, _, y1_test = train_test_split(
    X_1, y_1, test_size=10, random_state=42)

X_test = np.append(X1_test, X0_test)
y_test = np.append(y1_test, y0_test)

#%%
y_test

#%%
X_test
