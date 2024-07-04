# this python file if for exercise 2 question 3 

def dfs(current_node, graph, visited=None):
    """ Performs a depth-first search (dfs) on a graph starting from the current_node
    :param current_node: the node where the dfs starts
    :param graph: A dictionary representing the graph where keys are node names
    :param visited: A set to keep track of visited nodes. If None, it is initialised as an empty set within the function
    :return bool: True if the goal node "G" is found, False otherwise.
    """
    # initialise visited set if None
    if visited is None:
        visited = set()

    # Mark the current node as visited
    visited.add(current_node)

    # Check if the current not is the goal node
    if current_node == "G":
        print("Goal node found:", current_node)
        return True

    # Iterate over all neightbors of the current node
    for neighbors in graph[current_node]:
        # If the neighbor has not been visitied, perform DFS on the neighbor
        if neighbors not in visited: 
            if dfs(neighbors, graph, visited):
                return True

    # Return false if the goal node "G" is not found in this path 
    return False


# Graph definition
# each node is defined with their respective neighbors
graph = {
    "S": set(["d", "e", "p"]),
    "d": set(["b", "c", "e"]),
    "b": set(["a"]),
    "c": set(["a"]),
    "a": set([]),
    "f": set(["c", "G"]),
    "G": set([]),
    "e": set(["h", "r"]),
    "r": set(["f"]),
    "h": set(["p", "q"]),
    "p": set(["q"]),
    "q": set([])
}

if __name__ == "__main__":
    # Start DFS from the start node "S"
    dfs("S", graph)