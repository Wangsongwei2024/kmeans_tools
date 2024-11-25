import numpy as np
from kmeans_tools.CalDistance import caldistance


# 计算所有样本到初始质心的距离，并分类
def classsamples(X,samples,K,center):
    # 每个样本点到所有质心的距离
    dis_K = []
    # 所有样本的距离列表，每个元素就是每个样本点到所有质心的距离
    dis = []
    # 样本的分类列表
    classfier = []
    for i in range(samples):
        dis_K.clear()
        for j in range(K):
            d = caldistance(X[i], center[j])
            dis_K.append(d)

        # 按最小的距离，并归类
        min_value = min(dis_K)
        min_index = dis_K.index(min_value)
        classfier.append(min_index)
        dis.append(dis_K)
    return classfier