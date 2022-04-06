import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# 判斷是否分類錯誤
def check_error(w, x, y):
    x = np.array(x)
    if int(np.sign(w.T.dot(x))) != y:
        return True
    else:
        return False

# 計算分類錯誤率
def error_rate(w, dataset, num):
    error = 0.0
    for x, y in dataset:
        if check_error(w, x, y):
            error += 1.0
    return error / num

# Pocket Algorithm
def pla_pocket(dataset):
    # 初始化變數
    num = len(dataset)
    w = np.zeros(3)
    w_pocket = w
    error_pocket = error_rate(w, dataset, num)
    
    while True:
        rand_order = range(num)
        rand_order = random.sample(rand_order, num)

        finish = True       # 紀錄當前 w 是否已能正確分類
        for j in rand_order:
            # 隨機取分類錯誤的點更新 w
            x, y = dataset[j][0], dataset[j][1]
            if check_error(w, x, y):
                x = np.array(x)
                w = w + y * x
                finish = False
                
                # 計算新 w 的分類錯誤率是否小於 w_pocket 的
                error_wt = error_rate(w, dataset, num)
                if error_wt < error_pocket:
                    w_pocket = w
                    error_pocket = error_rate(w, dataset, num)
                    break
        if finish==True:
            break

    return w_pocket


if __name__ == '__main__':

    # 讀取樣本點資料
    data = pd.read_csv('./sample.csv', header=0)
    data = data.astype(int)
    x_coord = data.iloc[:,0]
    y_coord = data.iloc[:,1]
    labels = data.iloc[:,2]

    # 整理訓練資料, 格式為 [[(x0, x1, x2), y], ... ]
    b = []
    for i in range(len(labels)):
        b.append([(1, x_coord[i], y_coord[i]), labels[i]])
    dataset = np.array(b, dtype=object)

    # 執行 Pocket Algorithm
    w = pla_pocket(dataset)

    # 畫出結果圖
    point_n = int(len(labels) / 2)
    plt.plot(x_coord[:point_n], y_coord[:point_n], 'o', color='blue', markersize=2)   # positive
    plt.plot(x_coord[point_n:], y_coord[point_n:], 'x', color='red', markersize=2)    # negative
    
    x = np.arange(1000 + 1)
    a, b = -w[1]/w[2], -w[0]/w[2]
    y = a * x + b
    plt.plot(x, y)
    plt.show()

