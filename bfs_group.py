import os
import numpy as np
import sys
sys.setrecursionlimit(1000000)

def bfs_clustering(table, neighbor, ig_cls=0, show=False):
    h, w = table.shape
    history = np.zeros_like(table)

    # init group_res
    group_res = {}
    num_cls = np.unique(table)
    for i in num_cls:
        if i == ig_cls:
            continue
        group_res[i] = []

    def bfs(li, table, group_res):
        if len(li) == 0:
            return
        i, j = li.pop(0)
        cls = table[i, j]
        for dx in neighbor:
            i_ = i + dx
            if i_ >=0 and i_ < h:
                for dy in neighbor:
                    j_ = j + dy
                    if j_ >=0 and j_ < w:
                        if table[i_, j_] == cls and history[i_,j_] == 0:
                            history[i_,j_] = 1
                            group_res[cls][-1].append((i_,j_))
                            li.append((i_, j_))
        bfs(li, table, group_res)

    for cls in group_res:
        I, J = np.where(table==cls)
        all_i_j = [(i, j) for i, j in zip(I, J)]
        while all_i_j:
            i, j = all_i_j.pop(0)
            if history[i, j]:
                continue
            else:
                history[i, j] = 1
                group_res[cls].append([(i, j)])
                bfs([(i,j)], table, group_res)

    if show:
        # show table
        print(table)        
        # show group result
        for cls in group_res:
            print('----cls %d----' %cls)
            for g in group_res[cls]:
                if len(g) < 40:
                    continue
                tmp = np.zeros_like(table)
                for i, j in g:
                    tmp[i,j] = cls
                print(tmp)
                import cv2
                import time
                cv2.imwrite('{}.jpg'.format(time.time()), 80*tmp.astype(np.uint8))
                time.sleep(0.5)
    return group_res


if __name__ == "__main__":
    table = np.random.randint(0, 3, size=(10,10))
    neighbor = [1,]
    neighbor.extend([-i for i in neighbor])
    neighbor.append(0)
    group_res = bfs_clustering(table, neighbor, show=True)