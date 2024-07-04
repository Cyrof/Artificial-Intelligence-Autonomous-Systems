import random 

class NQueens:
    def __init__(self, n):
        """
        Initalise the NQueens board with n queens randomly placed. 
        :param n: the size of the board (n x n).
        """
        self.n = n
        self.board, self.queenPositions = self.getNewBoard(n)
          
    def getNewBoard(self, n):
        """
        create a new n x n baord and randomly place n queens on it
        :param n: the size of the board (n x n)
        :return tuple: a tuple containing the board and the positions of the queens.
        """
        board = []
        queenPos = []
        for x in range(n):
            board.append([0]*n)

        for x in range(n):
            randomIndex = random.randint(0, n-1)
            board[x][randomIndex] = 1
            queenPos.append((x, randomIndex))            

        return board, queenPos

    def allQueensSafe(self):
        """
        check if all queens are placed such that none of them can attack each other 
        :param:
        :return bool: True if all queens are safe, false otherwise 
        """
        return all(not self.underAttack(pos) for pos in self.queenPositions)

    def underAttack(self, pos):
        """
        check if a queen at a given position is under attack
        :param pos: the position of the queen (row, col)
        :return bool: true is the queen in under attack, false otherwise 
        """
        return self.attackViaDiagonal(pos) or self.attackViaRow(pos) or self.attackViaCol(pos)

    def attackViaCol(self, pos):
        """
        check if a queen at a given position is under attack via column 
        :param pos: the position of the queen (row, col)
        return bool: true if the queen is under attack via column, false otherwise
        """
        return any(pos[1] == queen[1] and queen != pos for queen in self. queenPositions)

    def attackViaRow(self, pos):
        """
        check if a queen at a given position is under attack via row 
        :param pos: the position of the queen (row, col)
        :return bool: truw if the queen is under attack via row, false otherwise 
        """
        return any(pos[0] == queen[0] and queen != pos for queen in self.queenPositions)
    
    def attackViaDiagonal(self, pos):
        """
        check if a queen at a given position is under attack via diagonal
        :param pos: the position of the queen (row, col)
        :return bool: true is the queen is under attack via diagonal, false otherwise
        """
        return any(abs(queen[0] - pos[0]) == abs(queen[1] - pos[1]) and queen != pos for queen in self.queenPositions)

    def availablePositions(self, row):
        """
        get all possible positions in a given row.
        :param row: the row number
        :return list: a list of tuples representing the positions in the row
        """
        return [(row, x) for x in range(self.n)]
    
    def moveQueen(self, startPos, endPos):
        """
        move a queen from one position to another
        :param startPos: The starting position of the queen (row, col)
        :param endPos: The ending position of the queen (row, col)
        :return:
        """
        assert self.board[startPos[0]][startPos[1]] == 1 # no queen at tthe starting position 
        self.board[startPos[0]][startPos[1]] = 0 
        self.board[endPos[0]][endPos[1]] = 1
        self.queenPositions.remove(startPos)
        self.queenPositions.append(endPos)
        
    def specificQueenConflicts(self, pos):
        """
        Count the number of conflicts for a queen at a given position 
        :param pos: the position of the queen (row, col)
        :return int: the number of conflicts for the queen
        """
        count = 0
        for queen in self.queenPositions:
            if abs(queen[0] - pos[0]) == abs(queen[1] - pos[1]) and queen != pos:
                count += 1
            if pos[0] == queen[0] and queen != pos: 
                count += 1
            if pos[1] == queen[1] and queen != pos: 
                count+= 1
        return count
    
    def printBoard(self):
        """
        Print the current state of the board.
        """
        for row in self.board:
            print(" ".join(str(cell) for cell in row))
        print()

