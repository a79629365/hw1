import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def gen_samples(m, b, num, rand_param):
    x_coors, y_coors, labels = np.array([]), np.array([]), np.array([])
    c = 1 if m >= 0 else -1

    n = int(num/2)

    for state in ['pos', 'neg']:
        x = np.random.randint(0, rand_param, n)
        r = np.random.randint(1, rand_param, n)

        if state == 'pos':
            y = m * x + b - (r * c)
            labels = np.append(labels, np.ones(n, dtype=int))
        else:
            y = m * x + b + (r * c)
            labels = np.append(labels, -1*np.ones(n, dtype=int))

        
        x_coors = np.append(x_coors, x)
        y_coors = np.append(y_coors, y)

    return x_coors, y_coors, labels

if __name__ == '__main__':
    # y = mx + b
    m, b = 2, 1
    m = int(input("請輸入m: "))
    b = int(input("請輸入b: "))
    num = int(input("請輸入sample points的數量: "))

    # 參數
    # num = 30                # 產生的樣本點總數
    rand_param = 1000         # 樣本點分布的範圍
    pos_num = int(num / 2)    # positive(negative) 樣本的數量

    # 根據 m, b 畫出 y = mx + b 之直線
    x = np.arange(rand_param + 1)   # x = [0, 1,..., rand_param]
    y = m * x + b
    plt.plot(x, y, linewidth=1)

    # 隨機產生樣本點
    x_coors, y_coors, labels = gen_samples(m, b, num, rand_param)

    # 將產生的樣本點和 label 輸出成 csv 檔
    dict = {'x': x_coors, 'y': y_coors, 'label': labels} 
    df = pd.DataFrame(dict) 
    df.to_csv('sample.csv', index=0)
    
    # 畫出產生的樣本點
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue', markersize=2)   # 藍色: positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'x', color='red', markersize=2)    # 紅色: negative
    plt.show()

    