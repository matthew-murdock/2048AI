import numpy as np


class Game():
    def __init__(self, size=4, invalid_move_penalty=0):
        self.size = size
        self.invalid_move_penalty = invalid_move_penalty
        self.score = 0
        self.board = np.zeros(size, dtype=int)

    def reset(self):
        """
        Sets the board to all zeros and then add two random tiles. Sets the score to zero.
        """
        pass

    def clear(self):
        """
        Sets the board to be all zeros and sets the score to zero.
        """
        pass

    def set_state(self, board, score):
        """
        Set the board and score to the values provided.

        Inputs:
        board (numpy.ndarray): the array in which the board will be set as (will first be copied)
        score (int): the value to set the score as
        """
        pass

    def set_board(self, board):
        """
        Set the board to the value provided.

        Inputs:
        board (numpy.ndarray): the array in which the board will be set as (will first be copied)
        """
        pass

    def set_score(self, score):
        """
        Set the score to the value provided.

        Inputs:
        score (int): the value to set the score as
        """
        pass

    def get_new_tile_options(self):
        """
        Generate options for a new tile's location, value a probability.

        Returns:
        indexes (numpy.ndarray): 2D array (n_options, 2) with row and column index for each 
                                 potential new tile
        values (numpy.ndarray): 1D array (n_options) with the tile value for each option
        probabilities (numpy.ndarray): 1D array (n_options) with the probability of each option
                                       being selected
        """
        pass

    def spawn_tile(self):
        """
        Add a random new tile (either 2 or 4) to a randomly selected open location.
        """
        pass

    def swipe(self, action):
        """"
        Execute a swipe in the direction that is provided.

        Inputs:
        action (int): the direction in to swipe

        Returns:
        points (int): the number of points earned on this swipe
        """
        pass

    def swipe_left(self):
        """
        Execute a swipe to the left. This is the default swiping direction and all other
        swiping directions will result in the board being transformed to execute the swipe
        in this direction.

        Returns:
        points (int): the number of points earned on this swipe
        """
        M, N = self.shape
        points = 0
        target_index = np.zeros(M, dtype=int)
        arange_index = np.arange(M)
        for n in range(1, N):
            nonempty_target_mask = self.board[arange_index, target_index] != 0
            sum_mask = (self.board[arange_index, target_index] == self.board[arange_index,n]) * nonempty_target_mask
            self.board[arange_index, target_index] += self.board[:,n] * sum_mask
            points += np.sum(self.board[arange_index, target_index] * sum_mask)
            self.board[arange_index,n] *= 1 - sum_mask
            target_index += sum_mask

            slide_mask = (1 - sum_mask) * (self.board[arange_index, n] != 0)
            target_index += slide_mask * (self.board[arange_index, target_index] != 0)
            self.board[arange_index, target_index] += self.board[arange_index, n] * slide_mask
            self.board[arange_index, n] *= 1 - slide_mask
        return points
    
    def step(self, action, return_distribution=False):
        """
        Execute a step in the provided direction. A step consists of a swipe and the spawn of a new tile.
        If the return_distritbution parameter is True, the method will return all possible next states
        as well as the probability of each state being selected. These values will be returned as a tuple
        in place of the single samples next state that would have otherwise been returned.

        Inputs:
        action (int): direction in which to swipe
        return_distributions (bool): Flag to request either a single sample of the next state (this is the 
                                     behavior when True) or a distribution over the next state (this is the
                                     behavior when False)

        Returns:
        next_state (numpy.ndarray): a copy of the new board that is generated. If return_distribution is false,
                                    this will be a 2-tuple. The first value will be a 3D array of all possible
                                    new boards. The second value will be a 1D array with the probability of
                                    each new board being selected.
        points (int): the number of points that were earned in this step
        game_over (bool): True is the game is over (i.e. there are no valid moves in the new state), False otherwise
        """
        pass

    def check_loss(self):
        """"
        Checks if the game has been lost (i.e. there are no valid moves remaining.)

        Returns:
        is_loss (bool): True is the game is lost, false otherwise
        """
        pass

    def get_state(self):
        """
        Returns the state of the game which includes the board and the score.

        Returns:
        board (numpy.ndarray): a copy of the current board
        score (int): the current score of the game
        """
        pass
    
    def get_board(self):
        """
        Returns the current board.
        
        Returns:
        board (numpy.ndarray): a copy of the current board
        """
        pass

    def get_score(self):
        """
        Returns the current score.

        Returns:
        score (int): the current score of the game
        """
        pass


class BoardTransformer():
    def __init__(self, board, action):
        self.board = board
        self.action = action

    def __enter__(self):
        pass

    def __exit__(self):
        pass