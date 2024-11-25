# K-MEANS 原理实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmeans_tools.CreSamples import cresamples
from kmeans_tools.ClassSamples import classsamples
from kmeans_tools.CalCenter import calcenter

np.random.seed(12)

# 首先创建随机样本点，及初始质心
samples = 50
ndim = 2
K = 3
X, init_center = cresamples(samples, ndim, K)

# 调用KMeans训练数据分类
KM = KMeans(n_clusters=K, init='random', random_state=42)
KM.fit(X)
labels_ = KM.labels_
KM_center = KM.cluster_centers_

# 按照初始质心分类
classfier = classsamples(X, samples, K,init_center)
print("center_init>>>>>>>")
print(init_center)
# 计算新质心
center_new = calcenter(X,samples,ndim,K,classfier)
print("center_new>>>>>>>>")
print(center_new)
classfier1 = classsamples(X, samples, K,center_new)


center_new1 = calcenter(X,samples,ndim,K,classfier1)
print("center_new1>>>>>>>>")
print(center_new1)
classfier2 = classsamples(X, samples, K,center_new1)

center_new2 = calcenter(X,samples,ndim,K,classfier2)
print("center_new2>>>>>>>>")
print(center_new2)
classfier3 = classsamples(X, samples, K,center_new2)

center_new3 = calcenter(X,samples,ndim,K,classfier3)
print("center_new3>>>>>>>>")
print(center_new3)
classfier4 = classsamples(X, samples, K,center_new3)

center_new4 = calcenter(X,samples,ndim,K,classfier4)
print("center_new4>>>>>>>>")
print(center_new4)
classfier5 = classsamples(X, samples, K,center_new4)

center_new5 = calcenter(X,samples,ndim,K,classfier5)
print("center_new5>>>>>>>>")
print(center_new5)
classfier6 = classsamples(X, samples, K,center_new5)

print("center_KM>>>>>>>>")
print(KM_center)


fig = plt.figure(figsize=(10,10))
fig_KMeans = plt.subplot(4,2,1)
plt.scatter(X[:, 0], X[:, 1], c=labels_, s=60)
plt.scatter(KM_center[:, 0], KM_center[:, 1], color='red', marker='s')
plt.xlabel('x')
plt.ylabel('y')
plt.title('KMeans')

fig_init = plt.subplot(4,2,2)
plt.scatter(X[:, 0], X[:, 1], c=classfier, s=60)
plt.scatter(init_center[:, 0], init_center[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('INIT-STATUS')

fig_1 = plt.subplot(4,2,3)
plt.scatter(X[:, 0], X[:, 1], c=classfier1, s=60)
plt.scatter(center_new[:, 0], center_new[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('The first class')

fig_2 = plt.subplot(4,2,4)
plt.scatter(X[:, 0], X[:, 1], c=classfier2, s=60)
plt.scatter(center_new1[:, 0], center_new1[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('The second class')

fig_3 = plt.subplot(4,2,5)
plt.scatter(X[:, 0], X[:, 1], c=classfier3, s=60)
plt.scatter(center_new2[:, 0], center_new2[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('The third class')

fig_4 = plt.subplot(4,2,6)
plt.scatter(X[:, 0], X[:, 1], c=classfier4, s=60)
plt.scatter(center_new3[:, 0], center_new3[:, 1], color='green', marker='d')
# plt.set_aspect("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.title('The fourth class')

fig_5 = plt.subplot(4,2,7)
plt.scatter(X[:, 0], X[:, 1], c=classfier5, s=60)
plt.scatter(center_new4[:, 0], center_new4[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('The fifth class')

fig_6 = plt.subplot(4,2,8)
plt.scatter(X[:, 0], X[:, 1], c=classfier6, s=60)
plt.scatter(center_new5[:, 0], center_new5[:, 1], color='green', marker='d')
plt.xlabel('x')
plt.ylabel('y')
plt.title('The sixth class')

plt.show()
