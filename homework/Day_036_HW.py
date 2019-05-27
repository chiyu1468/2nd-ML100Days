
#%% [markdown]
# ## 練習時間
# ### F1-Score 其實是 F-Score 中的 β 值為 1 的特例，代表 Precision 與 Recall 的權重相同
# 
# 請參考 F1-score 的[公式](https://en.wikipedia.org/wiki/F1_score) 與下圖的 F2-score 公式圖，試著寫出 F2-Score 的計算函數

#%% [markdown]
# HINT: 可使用 slearn.metrics 中的 precision, recall 函數幫忙
# > precision -> 判斷為 positive 裡面有多少是 真的為 positive
# > recall -> 真的為 positive 裡面有多少是 被判斷為 positive

#%%
import numpy as np
y_pred = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 prediction
y_true = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 ground truth

x = y_true - y_pred
Tp = np.count_nonzero(np.logical_and(y_pred, y_true))
Fn = np.count_nonzero(x > 0)
Fp = np.count_nonzero(x < 0)
Tn = len(x) - np.count_nonzero(np.logical_or(y_pred, y_true))

precision = Tp / (Tp+Fp)
recall = Tp / (Tp+Fn)

print(f'Tp : {Tp}, Fn : {Fn}, Fp : {Fp}, Tn : {Tn}')
print(f'precision : {precision:.4f}, recall : {recall:.4f}')

#%%
def FnScore(p, r, beta):
    return (1+beta**2)*p*r / ((beta**2)*p + r)


#%%
print(f'F1-score : {FnScore(precision, recall, 1):.4f}')
print(f'F2-score : {FnScore(precision, recall, 2):.4f}')


