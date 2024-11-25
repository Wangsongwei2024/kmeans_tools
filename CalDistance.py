import numpy as np

def caldistance(x1,x2):
    dis = 0
    if x1.shape[0] != x2.shape[0]:
        print(f"wrong position between x1[{x1.shape[0]}] and x2[{x2.shape[0]}]")
        return
    for i in range(x1.shape[0]):
        dis += (x1[i] - x2[i]) ** 2

    dis = np.sqrt(dis)
    return dis

if __name__ == '__main__':
    x1 = np.array([3,1,1])
    x2 = np.array([1,1,1])
    print(caldistance(x1,x2))