import numpy as np

### Evaluations score ###

piecesValues = {'k': 0, 'q': 10, 'r': 5, 'b': 3, 'n': 3, 'p': 1}
CHECKMATE = np.inf
STALEMATE = 0
DEPTH = 3

### Evaluation function ###

# Symmetric evaluation function: positive score means white is in a better 
# situation, and negative means black is in a better situation 

def board_eval(game):
    if game.checkmate:
        if game.white_move:
            return - CHECKMATE # black wins
        else:
            return CHECKMATE # white wins
    elif game.stalemate:
        return STALEMATE
    score = 0
    for row in game.board:
        for square in row:
            if square != '--':
                score += (2 * (square[0] == 'w') - 1) * piecesValues[square[1]]
    return score


### Random Agent ###

def randomMove(game, valid_moves):
    # Does not need the game argument, but added to match
    #  other AI moves parameters
    index = np.random.randint(0, len(valid_moves))
    return valid_moves[index]

### Greedy Agent ###

def greedyMove(game, valid_moves):
    turn = 2 * game.white_move - 1 #
    maxScore =  - CHECKMATE
    bestMove = None
    np.shuffle(valid_moves)
    for move in valid_moves:
        game.makeMove(move)
        #game.get_valid_moves()
        if game.checkmate:
            print('One move mates')
            score = CHECKMATE
        elif game.stalemate:
            score = STALEMATE
        else:
            score = board_eval(game) * turn
        if score >  maxScore:
            maxScore = score
            bestMove = move
        game.undoMove()
    if bestMove is None:
        bestMove = randomMove(game, valid_moves)
    return bestMove

### Minimax without recursion Agent ###

def minimaxWithoutRecMove(game, valid_moves):
    turn = 2 * game.white_move - 1 #
    opponentMinMaxScore =  CHECKMATE
    bestMove = None
    np.random.shuffle(valid_moves)
    for playerMove in valid_moves:
        game.makeMove(playerMove)
        opponentMoves = game.get_valid_moves()
        if game.stalemate:
            opponentMaxScore = STALEMATE
        elif game.checkmate:
            opponentMaxScore = - CHECKMATE
        else:
            opponentMaxScore = - CHECKMATE
            for opponentMove in opponentMoves:
                game.makeMove(opponentMove)
                game.get_valid_moves()
                if game.checkmate:
                    score = CHECKMATE
                elif game.stalemate:
                    score = STALEMATE
                else:
                    score = - board_eval(game) * turn
                if score > opponentMaxScore:
                    opponentMaxScore = score
                game.undoMove()
        if opponentMaxScore < opponentMinMaxScore:
            opponentMinMaxScore = opponentMaxScore
            bestMove = playerMove
        game.undoMove()
    if bestMove is None:
        bestMove = randomMove(game, valid_moves)
    return bestMove


def minimax (game, valid_moves):
    global nextMove
    nextMove = None
    np.random.shuffle(valid_moves)
    minimaxRecursion(game, valid_moves, DEPTH, game.white_move)
    if nextMove is None:
        nextMove = randomMove(game, valid_moves)
    return nextMove

def minimaxRecursion(game, valid_moves, depth, white_move):
    global nextMove
    if depth == 0:
        return board_eval(game)

    if white_move:
        maxScore = - CHECKMATE
        for move in valid_moves:
            game.makeMove(move)
            possibleMoves = game.get_valid_moves()
            score = minimaxRecursion(game, possibleMoves, depth - 1, False)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            game.undoMove()
        return maxScore

    else:
        minScore = CHECKMATE
        for move in valid_moves:
            game.makeMove(move)
            possibleMoves = game.get_valid_moves()
            score = minimaxRecursion(game, possibleMoves, depth - 1, True)
            if score < minScore:
                minScore = score
                if depth == DEPTH:
                    nextMove = move
            game.undoMove()
        return minScore


def negamax(game, valid_moves):
    global nextMove
    nextMove = None
    np.random.shuffle(valid_moves)
    turn = 1 if game.white_move else -1
    negamaxRecursion(game, valid_moves, DEPTH, turn)
    if nextMove is None:
        nextMove = randomMove(game, valid_moves)
    return nextMove

def negamaxRecursion(game, valid_moves, depth, turn):
    global nextMove
    if depth == 0:
        return turn * board_eval(game)
    
    maxScore = - CHECKMATE
    for move in valid_moves:
        game.makeMove(move)
        possibleMoves = game.get_valid_moves()
        score = - negamaxRecursion(game, possibleMoves, depth-1, -turn)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        game.undoMove()
    return maxScore


def negamaxAlphaBeta(game, valid_moves, depth, turn):
    global nextMove
    nextMove = None
    np.random.shuffle(valid_moves)
    turn = 1 if game.white_move else -1
    alpha = - CHECKMATE
    beta = CHECKMATE
    negamaxAlphaBetaRecursion(game, valid_moves, DEPTH, turn, alpha, beta)
    if nextMove is None:
        nextMove = randomMove(game, valid_moves)
    return nextMove

def negamaxAlphaBetaRecursion(game, valid_moves, depth, turn, alpha, beta):
    global nextMove
    if depth == 0:
        return turn * board_eval(game)
    
    # move ordering later
    maxScore = - CHECKMATE
    for move in valid_moves:
        game.makeMove(move)
        possibleMoves = game.get_valid_moves()
        score = - negamaxRecursion(game, possibleMoves, depth-1, -turn,
                                   -beta, -alpha)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        game.undoMove()
        if maxScore > alpha:
            alpha = maxScore
        if alpha >= beta:
            break
    return maxScore

