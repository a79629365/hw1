import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# 判斷是否分類錯誤
def check_error(w, x, y):
    x = np.array(x)
    if int(np.sign(w.T.dot(x))) != y:
        return True
    else:
        return False

# 計算分類錯誤數
def count_error(w, dataset):
    error = 0.0
    for x, y in dataset:
        if check_error(w, x, y):
            error += 1.0
    return error

# Pocket Algorithm
def pla_pocket(dataset, iter):
    # 初始化變數
    num = len(dataset)
    w = np.zeros(3)
    w_pocket = w
    error_pocket = count_error(w, dataset)
    update = 0
    
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
                update += 1 
                finish = False
                
                # 計算新 w 的分類錯誤數是否小於 w_pocket 的
                error_wt = count_error(w, dataset)
                if error_wt < error_pocket:
                    w_pocket = w
                    error_pocket = count_error(w, dataset)
                    break
            if update == iter:
                break

        if finish==True:
            break
        if update == iter:
            break

    return w_pocket, error_pocket


if __name__ == '__main__':

    # 讀取樣本點資料
    data = pd.read_csv('./sample_q4.csv', header=0)
    data = data.astype(int)
    x_coord = data.iloc[:,0]
    y_coord = data.iloc[:,1]
    labels = data.iloc[:,2]

    # 整理訓練資料, 格式為 [[(x0, x1, x2), y], ... ]
    b = []
    for i in range(len(labels)):
        b.append([(1, x_coord[i], y_coord[i]), labels[i]])
    dataset = np.array(b, dtype=object)
    num = len(dataset)

    # 執行 Pocket Algorithm
    start = time.time()
    w, err = pla_pocket(dataset, 100)
    end = time.time()
    accuracy = float((num - err) / num) * 100.0
    print("accuracy: %f" % accuracy)
    print("執行時間：%.8f 秒" % (end - start))

    # 畫出結果圖
    point_n = int(len(labels) / 2)
    for i in range(num):
        if labels[i] == 1:
            plt.plot(x_coord[i], y_coord[i], 'o', color='blue', markersize=2)   # positive
        else:
            plt.plot(x_coord[i], y_coord[i], 'x', color='red', markersize=2)    # negative
    
    x = np.arange(1000 + 1)
    a, b = -w[1]/w[2], -w[0]/w[2]
    y = a * x + b
    plt.plot(x, y)
    plt.show()

