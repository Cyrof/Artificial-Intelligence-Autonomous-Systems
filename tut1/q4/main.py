from time import time
from BFS_search import breadth_first_search

# three example initial states
state=[[1, 3, 4,
        8, 6, 2,
        7, 0, 5],

       [2, 8, 1,
        0, 4, 3,
        7, 6, 5],

       [2, 8, 1,
        4, 6, 3,
        0, 7, 5]]

# start to search actions for each initial state
for i in range(0,3):
    t0=time()
    bfs=breadth_first_search(state[i])
    t1=time()-t0
    print('BFS:', bfs)
    print('time:',t1)
    print('------------------------------------------')