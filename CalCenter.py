import numpy as np

def calcenter(X,samples,ndim,K,classfier):
    center_K = {}
    for i in range(samples):
        if center_K.get(classfier[i]) == None:
            list_tmp = []
            for k in range(ndim):
                list_tmp.append(0.00)
            # 数量
            list_tmp.append(0)
            center_K[classfier[i]] = list_tmp
        dim_list = center_K[classfier[i]]
        dim_list[-1] += 1

        for j in range(ndim):
            dim_list[j] += X[i][j]
        center_K[classfier[i]] = dim_list

    center_new = []
    for i in range(K):
        num = center_K[i][-1]
        center_dim = []
        for j in range(ndim):
            center_K[i][j] = center_K[i][j] / num
            center_dim.append(center_K[i][j])
        center_new.append(center_dim)
    center_new = np.array(center_new)
    return center_new
