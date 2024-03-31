import numpy as np


class Game():
    def __init__(self, size=4, invalid_move_penalty=0):
        if type(size) == int:
            size = (size, size)
        self.size = size
        self.invalid_move_penalty = invalid_move_penalty
        self.score = 0
        self.board = np.zeros(size, dtype=int)

    def reset(self):
        """
        Sets the board to all zeros and then add two random tiles. Sets the score to zero.
        """
        self.clear()
        for _ in range(2):
            self.spawn_tile()

    def clear(self):
        """
        Sets the board to be all zeros and sets the score to zero.
        """
        self.board *= 0

    def set_state(self, board, score):
        """
        Set the board and score to the values provided.

        Inputs:
        board (numpy.ndarray): the array in which the board will be set as (will first be copied)
        score (int): the value to set the score as
        """
        self.set_board(board)
        self.set_score(score)

    def set_board(self, board):
        """
        Set the board to the value provided.

        Inputs:
        board (numpy.ndarray): the array in which the board will be set as (will first be copied)
        """
        self.board = board.copy()

    def set_score(self, score):
        """
        Set the score to the value provided.

        Inputs:
        score (int): the value to set the score as
        """
        self.score = score

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
        indexes = np.argwhere(self.board == 0)
        values = np.ones(indexes.shape[0], dtype=int) + 1
        probabilities = np.ones(indexes.shape[0]) * 0.8

        indexes = np.concatenate((indexes, indexes), axis=0)
        values = np.concatenate((values, values*2))
        probabilities = np.concatenate((probabilities, 1-probabilities))
        
        return indexes, values, probabilities

    def spawn_tile(self):
        """
        Add a random new tile (either 2 or 4) to a randomly selected open location.
        """
        indexes, values, probabilities = self.get_new_tile_options()
        selection_index = np.random.choice(np.arange(probabilities.shape[0]), p=probabilities)
        self.add_tile(indexes[selection_index,:], values[selection_index])

    def add_tile(self, index, value):
        """
        Add a tile to the board.

        Inputs:
        index (iterable (len-2)): row and column index for the tile to be placed
        value (int): value to be placed at the specified index
        """
        self.board[index[0], index[1]] = value

    def swipe(self, action):
        """"
        Execute a swipe in the direction that is provided.

        Inputs:
        action (int): the direction in to swipe

        Returns:
        points (int): the number of points earned on this swipe
        """
        with BoardTransformer(self.board, action) as board:
            self.swipe_left()

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
        points = self.swipe(action)

        if return_distribution:
            indecies, values, probabilities = self.get_new_tile_options()
            M = probabilities.shape[0]
            next_boards = np.tile(np.expand_dims(self.board.copy(), 0), (M, 1, 1))
            next_boards[np.arange(M), indecies[:,0], indecies[:,1]] = values
            next_state = (next_boards, probabilities)
        else:
            self.spawn_tile()
            next_state = self.board.copy()
        
        return next_state, points, self.check_loss()

    def check_loss(self):
        """"
        Checks if the game has been lost (i.e. there are no valid moves remaining.)

        Returns:
        is_loss (bool): True is the game is lost, false otherwise
        """
        if np.sum(self.board == 0) != 0:
            is_loss = False
        else:
            match = np.any(self.board[:,:-1] == self.board[:,1:])
            match = match or np.any(self.board[:-1,:] == self.board[1:,:])
            is_loss = not match
        return is_loss

    def get_state(self):
        """
        Returns the state of the game which includes the board and the score.

        Returns:
        board (numpy.ndarray): a copy of the current board
        score (int): the current score of the game
        """
        return self.get_board(), self.get_score()
    
    def get_board(self):
        """
        Returns the current board.
        
        Returns:
        board (numpy.ndarray): a copy of the current board
        """
        return self.board.copy()

    def get_score(self):
        """
        Returns the current score.

        Returns:
        score (int): the current score of the game
        """
        return self.score


class BoardTransformer():
    def __init__(self, board, action):
        self.board = board
        self.action = action

    def __enter__(self):
        """
        Orient board so that a left swipe will result in the board being updated
        appropriately for the provided action.
        """
        if self.action == 2 or self.action == 3:
            self.board = np.transpose(self.board)
        
        if self.action == 1 or self.action == 3:
            self.board = np.flip(self.board, axis=1)
        return self.board

    def __exit__(self):
        """
        Undo the orientation to return the board to it's original state.
        """
        if self.action == 1 or self.action == 3:
            self.board = np.flip(self.board, axis=1)

        if self.action == 2 or self.action == 3:
            self.board = np.transpose(self.board)