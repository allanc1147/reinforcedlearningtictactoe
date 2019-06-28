import numpy as np


# Represent a game of tic tac toe
class Game:

    # Initialize the game
    def __init__(self):
        self.board = [[-1.0] * 3, [-1.0] * 3, [-1.0] * 3]

    # Reset the board
    def reset(self):
        self.board = [[-1.0] * 3, [-1.0] * 3, [-1.0] * 3]
        return self.board

    # Play 1 move of the game
    # -100 if scored out of bounds, -10 if cheating, 10 if tie, 100 if win
    # returns (self.board, reward, over, redo)
    def step(self, action, icon):
        if 0 > action[0] or 2 < action[0] or 0 > action[1] or 2 < action[1]:
            return self.board, -100, False, True
        if not self.is_space_free(action):
            return self.board, -10, False, True
        self.make_move(action, icon)
        if self.is_winner(icon):
            return self.board, 100, True, False
        if self.is_board_full():
            return self.board, 10, True, False
        return self.board, 0, False, False

    # Return true if there is a winning combo for the board
    def is_winner(self, icon):
        # check column
        for x in range(3):
            if self.board[x][0] == self.board[x][1] == self.board[x][2] == icon:
                return True
        # check row
        for y in range(3):
            if self.board[0][y] == self.board[1][y] == self.board[2][y] == icon:
                return True
        # check diagonal
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == icon:
            return True
        if self.board[2][0] == self.board[1][1] == self.board[0][2] == icon:
            return True
        return False

    # returns true if action is already occupied
    def is_space_free(self, action):
        return self.board[action[0]][action[1]] == -1.0

    # returns true if board is full
    def is_board_full(self):
        for x in range(3):
            for y in range(3):
                if self.is_space_free((x, y)):
                    return False
        return True

    # set the icon to coordinate action
    def make_move(self, action, icon):
        self.board[action[0]][action[1]] = icon


# Play tic tac toe as 2 human players
if __name__ == "__main__":

    def format_cell(x):
        if len(str(x)) == 1:
            return x + '\t'
        return x

    game = Game()
    done = False
    while not done:
        print("Move: <x y icon>")
        x, y, icon = input().split()
        board, reward, done, redo = game.step((int(x), int(y)), icon)
        print("Board")
        for y in range(3):
            print(format_cell(board[0][y]), '\t', format_cell(board[1][y]), '\t', format_cell(board[2][y]))
        print("Reward: ", reward)
        print("Finished game: ", done)
        print("Redo from invalid turn: ", redo)
        print("\n")
