import random
from NQueen.NQueens import NQueens

def nqueen_solve():
    # Solve the N-Queens problem using the min-conflict algorithm
    n = 8
    NQ = NQueens(n)
    max_steps = 1000 # Limit the numbmer of steps to avoid infinite loops
    
    for step in range(max_steps):
        # check if all queens are safe
        if NQ.allQueensSafe():
            print(f"Solved in {step} steps")
            NQ.printBoard()
            return 
        
        # Get a list of queens with conflicts 
        conflicts = [(q, NQ.specificQueenConflicts(q)) for q in NQ.queenPositions]  
        conflicts = [q for q in conflicts if q[1] > 0]
        if not conflicts:
            break # all queens are safe
        
        # randomly select a queen with conflicts 
        queen = random.choice(conflicts)[0]
        
        # Find the position that minimises conflicts 
        min_conflict_pos = queen 
        min_conflict = NQ.n + 1
        for pos in NQ.availablePositions(queen[0]):
            if pos != queen: 
                # move the queen to the new position temporarily
                NQ.moveQueen(queen, pos)
                conflicts = NQ.specificQueenConflicts(pos)
                
                # check if this position has fewer conflicts
                if conflicts < min_conflict:
                    min_conflict_pos = pos
                    min_conflict = conflicts
                    
                # move queen back to original position     
                NQ.moveQueen(pos, queen) 
            
        # move the queen to the position with the minimun conflicts 
        NQ.moveQueen(queen, min_conflict_pos)
    
    # if the loop exits without solving, print the failure message and board
    print("Failed to solve")
    NQ.printBoard()


if __name__ == "__main__":
    nqueen_solve()
