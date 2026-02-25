import numpy as np


head_pv = np.array([[1,2],[2,3]])
tail_pv = np.array([[4,5],[6,7]])
# 创建一维数组后重塑为列向量（shape=(n,1)）
waypoints = np.array([8,9])

print(head_pv[1])
print(tail_pv)
print(waypoints)
print('waypoints shape:', waypoints.shape)