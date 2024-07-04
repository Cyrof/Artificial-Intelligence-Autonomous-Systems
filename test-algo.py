# This python file is to play around with gym library to be used in the first assignment 

import gymnasium as gym
from collections import deque
import numpy as np
import sys

# Create the Taxi-v3 environment with text rendering mode 
env = gym.make("Taxi-v3", render_mode="ansi")
# env.reset(seed=2)
env.reset() # Reset the environment to its initial state 
        
def dfs(env):
    """
    Function to perform depth first search to solve the taxi-v3 environment 
    :param env: the environment to solve 
    :return: 
    """

    # Define possible actions the environment can do
    actions = [0, 1, 2, 3, 4, 5]  # Actions: South, North, East, West, Pickup, Dropoff
    stack = []  # Initilise the stack for DFS
    visited = set() #   Set to keep track of visited states

    # Get the initial state and add it to the stack
    initial_state = env.reset()[0]
    stack.append((initial_state, [], 0, False)) #   (state, path, accumulated reward, pickup status)  
    visited.add(initial_state) # Mark the intial state as visited

    while stack:    # Continue while there are states to process
        # pop the top state from the stack 
        state, path, total_reward, picked_up = stack.pop()
        env.unwrapped.s = state # set the environment to the surrent state 
        print(env.render()) # render the environment 

        # iterate over all possible actions 
        for action in actions:
            env.unwrapped.s = state  # Reset to the current state before taking an action 
            # take the action and get the result 
            next_state, reward, done, truncated, info = env.step(action)
            next_path = path + [action] # update the path with the current action 
            next_total_reward = total_reward + reward   # update the total reward

            # if the next state hasn't been visited, process it 
            if next_state not in visited:
                next_picked_up = picked_up  # copy the current pickup status
                # Decode the next state to ge the taxi and passenger information
                taxi_row, taxi_col, passenger_loc, destination = tuple(env.unwrapped.decode(next_state))

                # Check if passenger is picked up and  passenger is at a valid location 
                if action == 4 and passenger_loc != 4:  # Pickup action
                    if (taxi_row, taxi_col) == env.unwrapped.locs[passenger_loc]: # check if taxi is at passenger's location 
                        next_picked_up = True   # update pickup status

                # Check if the action is dropoff and the pasenger has been picked up 
                if action == 5 and picked_up:  # Dropoff action
                    if (taxi_row, taxi_col) == env.unwrapped.locs[destination]: # check if taxi is at the destination 
                        next_picked_up = False  # Passenger dropped off 
                        done = True  # Goal achieved

                # add the next state to visited and push it onto the stack 
                visited.add(next_state)
                stack.append((next_state, next_path, next_total_reward, next_picked_up))

            # If the goal state is reached, print the results and exit 
            if done:
                print("Goal reached!")
                print(env.render())
                print(f"Actions to reach goal: {next_path}")
                print(f"Total reward: {next_total_reward}")
                return

def bfs(env):
    queue = []
    visited = [False for _ in range(env.unwrapped.observation_space.n)]

    initial_state = env.reset()[0]
    queue.append((initial_state, [], 0, False))
    visited[initial_state] = True

    actions = np.array([0,1,2,3,4,5])

    while queue:
        state, path, total_reward, picked_up = queue.pop()
        env.unwrapped.s = state
        print(env.render())

        # for x in range(6):
            # env.unwrapped.s = state
        _, _, _, _, info = env.step(0)
        filtered_actions = actions[info['action_mask'] == 1]

        for action in filtered_actions:
            env.unwrapped.s = state
            next_state, reward, done, move, info = env.step(action)

            next_path = path + [action]
            new_total_reward = total_reward + reward

            if not visited[next_state]:
                new_picked_up = picked_up
                trow, tcol, ploc, dest = tuple(env.unwrapped.decode(next_state))

                if action == 4 and ploc!= 4:
                    if (trow, tcol) == env.unwrapped.locs[ploc]:
                        new_picked_up = True
                
                if action == 5 and picked_up:
                    if (trow, tcol) == env.unwrapped.locs[dest]:
                        new_picked_up = False
                        done = True
                
                visited[next_state] = True
                queue.append((next_state, next_path, new_total_reward, new_picked_up))
            
            if done: 
                print("Goal reached")
        
        print("no solution found")


if __name__ == "__main__":
    # dfs(env) # executes the dfs function 
    bfs(env)