#coding=utf-8
import xgboost as xgb
import pandas as pd
import time
import os
# import numpy as np
from sklearn.model_selection import train_test_split
from dataProcess import processHour,mergeTabel,mergeTabel1

processHour("train.csv")
trainDF = mergeTabel()

processHour("test.csv")
testDF = mergeTabel1()


userID = testDF["TERMINALNO"]

y = trainDF["Y"]
X = trainDF.drop(["TERMINALNO","Y"],axis=1)
test_X = testDF.drop(["TERMINALNO"],axis=1)

train_X,val_X, train_y, val_y = train_test_split(X,y,test_size = 0.2,random_state = 1)

num_trees = 450
# num_trees=45
params = {"objective": "reg:linear",
          "gamma":0.2,
          "lambda":2,
          "eta": 0.15,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

xgtest = xgb.DMatrix(test_X)

# 划分训练集与验证集
xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(val_X, label=val_y)

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(params, xgtrain, num_trees, evals=watchlist, early_stopping_rounds=200)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
preds = model.predict(xgtest,ntree_limit=model.best_iteration)
preds = [pred if pred >= 0 else 0 for pred in preds]
result = pd.DataFrame(columns=['Id',"Pred"])
result["Id"] = userID
result["Pred"] = preds
if "model" not in os.listdir("./"):
    os.mkdir("model")
result.to_csv("model/result.csv",index=False)
