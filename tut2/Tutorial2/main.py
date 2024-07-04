from time import time
from Astar_search import Astar_search
from puzzle import Puzzle

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

for i in range(0,3):
        t0 = time()
        astar = Astar_search(state[i])
        t1 = time() - t0
        print('A*:',astar)
        print('time:', t1)
        print()