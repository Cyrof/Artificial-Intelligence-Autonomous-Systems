class Puzzle:
    goal_state=[1,2,3,8,0,4,7,6,5] # I change the goal state to reduce the searching time!
    def __init__(self,state,parent,action,path_cost):
        self.parent=parent
        self.state=state
        self.action=action
        if parent:
            self.path_cost = parent.path_cost + path_cost
        else:
            self.path_cost = path_cost

    def __str__(self):
        # print the board 3x3
        return str(self.state[0:3])+'\n'+str(self.state[3:6])+'\n'+str(self.state[6:9])

    def goal_test(self):
        # TO DO
        # hint: check if current state is the goal state
        return self.state == Puzzle.goal_state

    def find_legal_actions(self, i,j):
        legal_action = ['U', 'D', 'L', 'R']
        if i == 0:  # up is disable
            legal_action.remove('U')
        elif i == 2:  # down is disable
            legal_action.remove('D')
        if j == 0: # left is disable
            legal_action.remove('L')
        elif j == 2: # right is disable
            legal_action.remove('R')
        return legal_action

    def generate_child(self):
        children=[]
        # get the location of 0 because only 0's neighbors can be moved
        x = self.state.index(0) 
        i = int(x / 3)
        j = int(x % 3)
        legal_actions=self.find_legal_actions(i,j)

        for action in legal_actions:
            new_state = self.state.copy()
            if action == 'U':
                new_state[x], new_state[x-3] = new_state[x-3], new_state[x]
            elif action == 'D':
                # TO DO
                new_state[x] , new_state[x+3] = new_state[x+3], new_state[x]
                pass
            elif action == 'L':
                new_state[x], new_state[x-1] = new_state[x-1], new_state[x]
            elif action == 'R':
                # TO DO
                new_state[x], new_state[x+1] = new_state[x+1], new_state[x]
                pass
            # TO DO
            # hint: append a new state to children
            if self.is_valid(new_state):
                children.append(Puzzle(new_state, self, action, 1))

                
        return children

    def is_valid(self, state):
        # check if the new state is valid (no duplicate numbers)
        return len(set(state)) == len(state)

    def find_solution(self):
        solution = []
        solution.append(self.action)
        path = self
        while path.parent != None:
            path = path.parent
            solution.append(path.action)
        solution = solution[:-1]
        solution.reverse()
        return solution