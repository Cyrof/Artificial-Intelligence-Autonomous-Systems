class Puzzle:
    goal_state=[1,2,3,8,0,4,7,6,5]
    def __init__(self,state,parent,action,path_cost):
        self.parent=parent
        self.state=state
        self.action=action
        # g(x)
        if parent:
            self.path_cost = parent.path_cost + path_cost
        else:
            self.path_cost = path_cost
        # h(x)
        self.generate_heuristic()
        # f(x) = g(x) + h(x)
        self.evaluation_function=self.path_cost+self.heuristic

    def __str__(self):
        return str(self.state[0:3])+'\n'+str(self.state[3:6])+'\n'+str(self.state[6:9])

    def generate_heuristic(self):
        # TO DO
        self.heuristic = 0
        for i in range(1, 9):
            current_index = self.state.index(i)
            goal_index = self.goal_state.index(i)
            current_row, current_col = divmod(current_index, 3)
            goal_row, goal_col = divmod(goal_index, 3)
            self.heuristic += abs(current_row - goal_row) + abs(current_col - goal_col)


    def goal_test(self):
        return self.state == self.goal_state
    

    def find_legal_actions(self, i,j):
        legal_action = ['U', 'D', 'L', 'R']
        if i == 0:  # up is disable
            legal_action.remove('U')
        elif i == 2:  # down is disable
            legal_action.remove('D')
        if j == 0:
            legal_action.remove('L')
        elif j == 2:
            legal_action.remove('R')
        return legal_action

    def generate_child(self):
        children=[]
        x = self.state.index(0)
        i = int(x / 3)
        j = int(x % 3)
        legal_actions=self.find_legal_actions(i,j)

        for action in legal_actions:
            new_state = self.state.copy()
            if action == 'U':
                new_state[x], new_state[x-3] = new_state[x-3], new_state[x]
            elif action == 'D':
                new_state[x], new_state[x+3] = new_state[x+3], new_state[x]
            elif action == 'L':
                new_state[x], new_state[x-1] = new_state[x-1], new_state[x]
            elif action == 'R':
                new_state[x], new_state[x+1] = new_state[x+1], new_state[x]
            children.append(Puzzle(new_state,self,action,1))
        return children

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