
#%% [markdown]
# # 作業一
# https://www.kaggle.com/daveianhickey/2000-16-traffic-flow-england-scotland-wales
# 1. 你選的這組資料為何重要
# * 分析哪邊是交通事故的熱區，可以對該區做事故防範。
# 2. 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)
# * 政府當局提供的 open data
# 3. 蒐集而來的資料型態為何
# * csv 表格結構化資料
# 4. 這組資料想解決的問題如何評估
# * 若該區事故率有波動，可以調查是否有道路或環境變更。

#%% [markdown]
# # 作業二
# 想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：
# 1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)
# * 首先定義業績：自由載客車隊服務次數。初期把單次服務獲利放在第二考量，以提升知名度為主。
# 2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)
# * 地域資料：同業狀況、客群分布、載客時間分布
# * 營運資料：目前車輛狀況及成本、服務時間與地點
# 3. 蒐集而來的資料型態為何
# * 結構化資料
# 4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)
# * 服務次數是否有提升趨勢

#%% [markdown]
# # 作業三

#%%
import numpy as np
import matplotlib.pyplot as plt



#%%
def mean_squared_error(y,yp):
    """
    請完成這個 Function 後往下執行
    """
    return np.mean(np.square(y - yp))



#%%
def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae


#%%
w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()



#%%
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()



#%%
# 執行 Function, 確認有沒有正常執行
MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))




#%%



