import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url) #讀取資料
print(data.head()) #列出前部分資料

data_x = data["YearsExperience"]
data_y = data["Salary"]
sum_x = np.sum(data_x) #x總和
sum_y = np.sum(data_y) #y總和
average_x = sum_x/len(data_x) #平均數x
average_y = sum_y/len(data_y) #平均數y

print("X數據的總和: ",sum_x)
print("Y數據的總和: ",sum_y)
print("X數據平均: ",average_x)
print("Y數據平均: ",average_y)

pre_x = np.sum((data_x - average_x)**2)
pre_y = np.sum((data_y - average_y)**2)
pre_xy = np.sum((data_x-average_x)*(data_y-average_y))

std_x = math.sqrt(pre_x/len(data_x)) #x的標準差
std_y = math.sqrt(pre_y/len(data_y)) #y的標準差
print("X,Y的標準差: ",std_x,std_y)
r_xy = pre_xy/(math.sqrt(pre_x)*math.sqrt(pre_y)) #x,y相關係數
print("X,Y的相關係數: ",r_xy)
m= r_xy*(std_y/std_x) #直線方程式的斜率
x_max = np.max(data_x)
x_min = np.min(data_x)
x1 = np.linspace(x_min,x_max,50)
y1 = m*(x1-average_x)+average_y # y = (r_xy*(std_y/std_x))*(x1-average_x)+average_y
plt.grid() 
plt.scatter(data_x,data_y) #數據的分布圖
plt.plot(x1,y1) #直線方程式圖形
plt.show()

