# import required libraries for this program
import gymnasium as gym
from copy import deepcopy
from collections import deque
import heapq
import time


class algo_test:
    """
    Class to test different path finding algorithms (BFS, UCS, A*) on the Taxi-v3 environment

    Methods
    -------
    __init__(self, env="Taxi-v3", render_mode="ansi"):
    Initializes the environment and sets up the required variables.
    display(self, path=[], reward=0, steps=0, time=0, algo_type="undefined", line_break=False):
        Displays the results of the algorithm execution.
    bfs(self):
        Executes the Breadth-First Search (BFS) algorithm on the environment.
    heuristic(self, done):
        Heuristic function for the A* search algorithm.
    afs(self):
        Executes the A* Search (AFS) algorithm on the environment.
    ucs(self):
        Executes the Uniform Cost Search (UCS) algorithm on the environment.
    """

    def __init__(self, env="Taxi-v3", render_mode="ansi"):
        """
        Initialise the algo_test class with the given environment and render mode.
        :param env: str, optional - The environment to use (default is "Taxi-v3")
        :param render_mode: str, optional - The mode in which to render the environment (default is "ansi")
        :raises Exeception: if there's any unknown error or the render mode is invalid
        :return: 
        """
        self.env = None
        try:
            # Attempt to create the environment with the given render mode  & env
            self.env = gym.make(env, render_mode=render_mode)
            self.env.reset()    # Reset the environment to the initial state
        except Exception as e:
            # Raise an error if there's an issue with environment creation or render mode
            raise f"Unknown error or render mode {e}"

        # Get the number of possible actions in the environment
        self.actions = self.env.unwrapped.action_space.n

    def display(self, path=[], reward=0, steps=0, time=0, algo_type="undefined", line_break=False):
        """
        Displays the results of the algorithm execution.
        :param path: list, optional - The path taken to reach the goal (default is []).
        :param reward: int, optional - The total reward accumulated (default is 0).
        :param steps: int, optional - The total number of steps taken (default is 0).
        :param time: float, optional - The total time taken (default is 0).
        :param algo_type: str, optional - The type of algorithm used (default is "undefined").
        :param line_break: bool, optional - Whether to print a line break after ths display (default is false).
        :return:
        """
        print(f"{algo_type} algorithm:")
        print("Goal reached!!")
        print(f"Total time: {time:.5f} ms")
        print(f"Total steps: {steps}")
        print(f"Path taken: {path}")
        print(f"Reward: {reward}")
        if line_break:
            print("\n")

    def bfs(self):
        """
        Executes the Breath-First Search (BFS) algorithm on the environment.

        This method uses BFS to find a path from the inital state to a goal state.
        It explores the state space level by level and stops when it reaches a goal state.

        The result are displayed using the display method.
        --------
        :param:
        :return:
        """
        start_time = time.time()    # Start the timer to measure execution time
        total_steps = 0  # Initialise total steps center
        copy_env = deepcopy(self.env)   # Create a deep copy of the environment
        initial_state, _ = copy_env.reset()  # Reset the environment to the initial state
        # Initialise the BFS queue with the initial state and empty path
        queue = deque([(initial_state, [])])
        visited = set()  # Initiliase the visited set to keep track of visited states

        visited.add(initial_state)  # Add the iniitial state to the visited set
        total_reward = 0    # Initialise the total reward counter
        while queue:
            # Get the current state and path from the front of the queue
            state, path = queue.popleft()

            for action in range(copy_env.unwrapped.action_space.n):
                copy_env.unwrapped.s = state    # Set the environment state to the current state
                # Perform the action and get the new state and reward
                new_state, reward, done, _, _ = copy_env.step(action)

                if new_state not in visited:
                    total_steps += 1    # Increment the total steps coujtner
                    # Add the new state to the visited set
                    visited.add(new_state)
                    # Append the new state and updated path to the queue
                    queue.append((new_state, path+[action]))
                    total_reward += reward  # Add the reward to the total reward

                if done:
                    # If the goal state is reached, display the results and return
                    self.display(path=path+[action], reward=total_reward, steps=total_steps, time=time.time(
                    )-start_time, algo_type="Breath first search", line_break=True)
                    return

        print("No solution found")  # print a message if no solution is found

    def heuristic(self, done):
        """
        Modified Heuristic function for the A* search algorithm
        --------
        :param done: bool - Whether the goal state has been reached.
        :returns 0 or 1: int - The heuristic value (0 if goal is reached, 1 otherwise).
        """
        if done:
            return 0    # Return 0 if goal state is reached
        return 1    # return 1 otherwise

    def afs(self):
        """
        Executes the A* Search (AFS) algorithm on the environment.

        This method uses the A* algorithm to find a path from the initial state to a goal state.
        It uses a priority queue to explore the state space, prioritising states with lower cost and heuristic values.

        The result are displayed using the display method.
        -------
        :param: 
        :return: 
        """
        start_time = time.time()    # start the timer to measure execution time
        total_steps = 0  # initialise total steps counter
        copy_env = deepcopy(self.env)   # create a deep copy of the environment

        initial_state, _ = copy_env.reset()  # reset the environment to the initial state
        # initialise the priority queue with the initial state, cost, and an empty path
        queue = [(0, initial_state, [])]
        visited = set()  # initialise the visited set to keep track of visited states

        visited.add(initial_state)  # add the initial state to the visited set
        # initialise the cost dictionary with the initial state and cost 0
        cost = {initial_state: 0}
        total_reward = 0    # initialise the total reward counter
        while queue:
            # get the state with the lowest cost and heuristic value from the queue
            _, state, path = heapq.heappop(queue)

            done = False
            for action in range(copy_env.unwrapped.action_space.n):
                copy_env.unwrapped.s = state    # set the environment state to the current state
                # perform the action and get the new state and reward
                new_state, reward, done, _, _ = copy_env.step(action)
                new_cost = cost[state] + 1  # Calculate the new cost

                if new_state not in visited or cost.get(new_state, float('inf')) > new_cost:
                    total_steps += 1    # Increment the total steps counter
                    # update the cost dictionary with the new state and cost
                    cost[new_state] = new_cost
                    # add the new state to the visited set
                    visited.add(new_state)
                    # calculate the priority using the new cost and heuristic
                    priority = new_cost + self.heuristic(done)
                    # Push the new statem, priority, and path to the queue
                    heapq.heappush(queue, (priority, new_state, path+[action]))
                    total_reward += reward  # add the reward to the total reward

                if done:
                    # if the goal state if reached, display the results and return
                    self.display(path=path+[action], reward=-total_reward, steps=total_steps,
                                 time=time.time()-start_time, algo_type="A* search", line_break=True)
                    return
        print("No solution found")  # Print a message if no solution is found

    def ucs(self):
        """
        Executes the Uniform Cost Search (UCS) algorithm on the environment.

        This method uses UCS to find a path from the initial state to a goal state. 
        It uses a priority queue to explore the state space, prioritising states with the lowest cumulative cost.

        The results are displayed using display method.
        -------
        :param:
        :return:
        """
        start_time = time.time()    # start the t imer to measure execution time
        total_steps = 0  # initialise total steps counter
        copy_env = deepcopy(self.env)   # create a deep copy of the environment

        initial_state, _ = copy_env.reset()  # reset the environment to the initial state
        # initialise the priority queue with the initial state, cost, and an empty path
        queue = [(0, initial_state, [])]
        visited = set()  # initialise the visited set to keep track of visited states

        visited.add(initial_state)  # add the initial state to the visited set
        # initialise the cost dictionary with the intial state and cost 0
        cost = {initial_state: 0}
        total_reward = 0    # initialise the total reward counter

        while queue:
            # get the state with lowest cumulative cost from the queue
            current_cost, state, path = heapq.heappop(queue)

            done = False
            for action in range(copy_env.unwrapped.action_space.n):
                copy_env.unwrapped.s = state    # set the environement state to the curren state
                # perform the action and ge the new state and reward
                new_state, reward, done, _, _ = copy_env.step(action)
                new_cost = current_cost + 1  # calculate the new cumulative cost

                if new_state not in visited or cost.get(new_state, float('inf')) > new_cost:
                    total_steps += 1    # increment the total steps counter
                    # update the cost dictionary with the new state and cost
                    cost[new_state] = new_cost
                    # add the new state to the visited set
                    visited.add(new_state)
                    heapq.heappush(
                        # Push the new state, cost, and path to the queue
                        queue, (new_cost, new_state, path + [action]))
                    total_reward += reward  # add the reward to the total reward

                if done:
                    # if the goal state is reached, display the results and return
                    self.display(path=path + [action], reward=-total_reward, steps=total_steps,
                                 time=time.time()-start_time, algo_type="Uniform cost search", line_break=True)
                    return
        print("No solution found")  # print a message if no solution is found


if __name__ == "__main__":
    a = algo_test()  # create an instance of the algo_test class
    a.bfs()  # execute the BFS algorithm
    a.ucs()  # execute the Uniform cost search algorithm
    a.afs()  # execute the A* search algorithm
