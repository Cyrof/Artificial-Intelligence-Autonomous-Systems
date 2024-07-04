from queue import Queue
from puzzle import Puzzle


def breadth_first_search(initial_state):
    start_node = Puzzle(initial_state, None, None, 0)
    if start_node.goal_test():
        return start_node.find_solution()
    q = Queue()
    q.put(start_node)
    explored=[]
    while not(q.empty()):
        # TO DO
        current_node = q.get() # Dequeue a node from the front of the queue
        explored.append(current_node.state) # add the current node's state to the explored list

        if current_node.goal_test(): # check if the current node is the goal
            return current_node.find_solution()
        
        # Enqueue all unexplored neighbors of the current node
        for child in current_node.generate_child():
            if child.state not in explored:
                q.put(child)

    return None # return None if no solution if found 