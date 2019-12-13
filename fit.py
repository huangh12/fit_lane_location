import numpy as np
import matplotlib.pyplot as plt
import cPickle
from easydict import EasyDict as edict
from matplotlib.pyplot import MultipleLocator
from bfs_group import bfs_clustering

config = edict()
config.minimum_points = 50
config.max_group = 3
config.max_neighbor_distance = 20

def indicator(x):
    x_square_sum, x_sum = np.sum(x**2), np.sum(x)
    det = len(x) * x_square_sum - x_sum**2
    return x_square_sum, x_sum, det


def solve_k_b(x, y):
    x_square_sum, x_sum, det = indicator(x)
    while det == 0:
        x = x[:-1]
        x_square_sum, x_sum, det = indicator(x)
    N_ = len(x)
    k_ = np.sum(y * (N_*x-x_sum)) / det
    b_ = np.sum(y * (x_square_sum-x*x_sum)) / det
    return N_, k_, b_

import os; os.system('rm ./*.jpg ./*.png' )

with open('img_all_res.pkl', 'rb') as f:
    img_list = cPickle.load(f)['all_seg_results']
    img_list = [_['seg_results'] for _ in img_list]

for cnt, img in enumerate(img_list):
    h, w = img.shape[:2]

    import cv2
    cv2.imwrite('raw%d.jpg' %cnt, 80*img)

    ax=plt.gca()
    plt.xlim(0, w)
    plt.ylim(0, h)    
    ax.set_aspect('equal', adjustable='box')

    # cluster the lane points
    neighbor = list(range(1, config.max_neighbor_distance+1))
    neighbor.extend([-i for i in neighbor])
    neighbor.append(0)
    group_res = bfs_clustering(img, neighbor, ig_cls=0, show=False)

    for cls in group_res:
        print('----cls %d----' %cls)
        for g in group_res[cls]:
            if len(g) < config.minimum_points:
                continue        
            print('group length: %d' %(len(g)))
            x, y = [], []
            for i, j in g:
                x.append(j)
                y.append(h-1-i)
            x = np.array(x, dtype='float32')
            y = np.array(y, dtype='float32')

            plt.scatter(x,y)

            N_, k_, b_ = solve_k_b(x, y)
            print(N_, k_, b_)
            # plot the fitted line
            xp = np.linspace(np.min(x), np.max(x), 200)
            yp = k_ * xp + b_
            plt.plot(xp, yp, color='black', linestyle='--' if cls==1 else '-')  
    plt.savefig('draw%d.png' %cnt)
    plt.clf()
