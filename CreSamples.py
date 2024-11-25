import numpy as np

def cresamples(samples,ndim,K):
    X = 10 * np.random.random(size=(samples, ndim))
    center = 10 * np.random.random(size=(K, ndim))
    return X,center

if __name__ == '__main__':
    X,init_center = cresamples(4,3,2)
    print("X>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(X)
    print("init_center>>>>>>>>>>>>>>>>")
    print(init_center)