from queue import PriorityQueue
from puzzle import Puzzle

def Astar_search(initial_state):
    '''
    In A* search, when multiple nodes have the same evaluation function value, 
    you need a way to determine the order in which they should be expanded. 
    This is where "count" comes into play.
    '''
    count=0
    explored=set()
    start_node=Puzzle(initial_state,None,None,0)
    q = PriorityQueue()
    q.put((start_node.evaluation_function,count,start_node))
   
    while not q.empty():
        node=q.get()
        print(node)
        node=node[2]
        explored.add(tuple(node.state))
        if node.goal_test():
            return node.find_solution()
        # TO DO
        explored.add(tuple(node.state))
        
        for child in node.generate_child():
            if tuple(child.state) not in explored:
                count += 1
                q.put((child.evaluation_function, count, child))
    
    return
