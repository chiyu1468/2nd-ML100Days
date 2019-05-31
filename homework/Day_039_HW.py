#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass


#%% [markdown]
# ## [作業重點]
# 清楚了解 L1, L2 的意義與差異為何，並了解 LASSO 與 Ridge 之間的差異與使用情境
#%% [markdown]
# ## 作業
#%% [markdown]
# 請閱讀相關文獻，並回答下列問題
# 
# [脊回歸 (Ridge Regression)](https://blog.csdn.net/daunxx/article/details/51578787)
# [Linear, Ridge, Lasso Regression 本質區別](https://www.zhihu.com/question/38121173)
# 
# 1. LASSO 回歸可以被用來作為 Feature selection 的工具，請了解 LASSO 模型為什麼可用來作 Feature selection
# 2. 當自變數 (X) 存在高度共線性時，Ridge Regression 可以處理這樣的問題嗎?
# 

#%% [markdown]
# 題目一
# Lasso Regression 
# the penalty function in LASSO, it's partial derivative will create +α / -α term, 
# could make the parameter of the feature down to zero.
# so we can use this characteristic to do feature selection.

#%% [markdown]
# 題目二
# 對於這個活動的參與學員，這題目設計並不好，大家並不知道何謂 collinearity (Multicollinearity)，也不清楚其會造成之影響。
# 可以，但是詳細原理還沒想清楚
# 參考資料: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf



