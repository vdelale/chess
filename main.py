"""
This will be the main file for running the chess game.
"""

import agents as ag
import engine as eng
import pygame as pg
import time


WIDTH = HEIGHT = 800
MOVE_LOG_HEIGHT = HEIGHT
MOVE_LOG_WIDTH = 3 * WIDTH // 8
DIMENSION = 8
SQUARE_SIZE = WIDTH / 8
MAX_FPS = 15  # For game animation
IMAGES = {}
COLORS = {'white': '#f0d9b5', 'black': '#b58863'}
ALGORITHMS = {None: ag.randomMove, 'greedy': ag.greedyMove,
              'basic_minimax': ag.minimaxWithoutRecMove, 'minimax': ag.minimax,
              'negamax': ag.negamax, 'negamax_alpha_beta': ag.negamaxAlphaBeta}
# Dictionnary containing all the different agents
UNDO = pg.K_BACKSPACE
RESET = pg.K_r
# Constants for keypress (to reduce lines length)
"""
We will initalize a global dictionnary of images used to represent the pieces.
"""


def load_images():
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    full_pieces = ['w' + piece for piece in pieces]
    full_pieces += ['b' + piece for piece in pieces]
    for piece in full_pieces:
        path = './pieces/' + piece + '.png'
        image = pg.image.load(path)
        IMAGES[piece] = pg.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))


def main(playerOne=False, playerTwo=False, algo1=None, algo2=None,
         animations=True):
    """
    Parameters:
        playerOne (boolean): whether the first player is human or not

        playerTwo (boolean): whether the second player is human or not

        algo1 (str): string corresponding to one of the keys of ALGORITHMS
                     dictionnary, dictates the behaviour of the first AI if
                     playerOne is False
        algo2 (str): same as algo1 for the second algorithm

        animations (boolean): whether or not to animate the moves
    """

    # Sets the board
    pg.init()
    screen = pg.display.set_mode((WIDTH + MOVE_LOG_WIDTH, HEIGHT))
    clock = pg.time.Clock()
    screen.fill(pg.Color("white"))
    load_images()
    LogFont = pg.font.SysFont('TimesNewRoman', 18, False, False)
    running = True
    selectedSquare = None   # No square selected at the beginning
    # will be a tuple (colum , row)
    player_clicks = []  # Keep track of the player clicks
    gameOver = False

    # Game initalisation

    game = eng.Game()
    valid_moves = game.get_valid_moves()
    move_made = False
    AI_1_move = ALGORITHMS[algo1]
    AI_2_move = ALGORITHMS[algo2]
    AI_1_turn = True
    # If human plays white True, else if AI, False
    # Same but for black

    # We wil write here some function to avoid code repetitions in the code
    def reset_game():
        nonlocal game, valid_moves, selectedSquare, player_clicks, move_made
        nonlocal gameOver, humanTurn, AI_1_turn
        game = eng.Game()
        valid_moves = game.get_valid_moves()
        selectedSquare = None
        player_clicks = []
        move_made = False
        gameOver = False
        humanTurn = (game.white_move and playerOne)
        humanTurn |= (not game.white_move and playerTwo)
        AI_1_turn = True

    def undo(playerOne, playerTwo):
        nonlocal game, move_made, animate, gameOver, humanTurn
        # We want to undo only one move if two human play
        # but two moves if only one player is human
        # since the AI plays directly without pause
        if (playerOne and playerTwo):
            game.undoMove()
            move_made = True
            animate = False
            gameOver = False
            humanTurn = (game.white_move and playerOne)
            humanTurn |= (not game.white_move and playerTwo)
        elif playerOne ^ playerTwo:  # xor : we want them different
            checkmate = game.checkmate
            game.undoMove()
            if not checkmate:
                game.undoMove()
            # If there is checkmate we undo only the move that mates
            # else, we need to undo both the player's and the AI's moves
            move_made = True
            animate = False
            gameOver = False
            humanTurn = (game.white_move and playerOne)
            humanTurn |= (not game.white_move and playerTwo)
        # else: two AIs therefore do not take into account

    def manageGameOver():
        nonlocal running, gameOver
        pg.event.set_blocked(None)
        pg.event.set_blocked(772)  # block unknown actions
        pg.event.set_allowed([pg.QUIT, pg.KEYDOWN])
        # Allow only mouse clicks, keys and window closing to be recognized
        e = pg.event.wait()
        if e.type == pg.QUIT:
            running = False
        elif e.type == pg.KEYDOWN and e.key in (UNDO, RESET):
            if e.key == RESET:
                reset_game()
            elif e.key == UNDO:
                undo(playerOne, playerTwo)

    # Beginning of the game

    while running:
        humanTurn = (game.white_move and playerOne)
        humanTurn |= (not game.white_move and playerTwo)
        if humanTurn:
            pg.event.set_blocked(None)
            pg.event.set_blocked(772)
            pg.event.set_allowed([pg.QUIT, pg.MOUSEBUTTONDOWN, pg.KEYDOWN])
            # Allow only mouse clicks, keys and window closing to be recognized
            e = pg.event.wait()
            # the wait function enables to pause the function while then
            # human player think of his move, rather than going the full loop
            if e.type == pg.QUIT:
                running = False
            # Mouse actions
            elif e.type == pg.MOUSEBUTTONDOWN:  # Square selection with mouse
                if not gameOver and humanTurn:
                    location = pg.mouse.get_pos()
                    col = int(location[0] // SQUARE_SIZE)
                    row = int(location[1] // SQUARE_SIZE)
                    if selectedSquare == (row, col) or col >= 8:
                        selectedSquare = None  # Deselect
                        player_clicks = []  # clear clicks
                    else:
                        selectedSquare = (row, col)
                        player_clicks.append(selectedSquare)
                    if len(player_clicks) == 2:
                        move = eng.Move(player_clicks[0], player_clicks[1],
                                        game)
                        for i in range(len(valid_moves)):
                            if move == valid_moves[i]:
                                game.makeMove(valid_moves[i])
                                move_made = True
                                animate = animations
                                selectedSquare = None
                                player_clicks = []
                        if not move_made:
                            player_clicks = [selectedSquare]

            # Keyboard actions

            elif e.type == pg.KEYDOWN:
                if e.key == UNDO:
                    undo(playerOne, playerTwo)
                if e.key == pg.K_h:
                    print([str(move) for move in valid_moves])
                if e.key == RESET:
                    reset_game()

        if not gameOver and not humanTurn:
            if AI_1_turn:
                AI_move = AI_1_move(game, valid_moves)
            else:
                AI_move = AI_2_move(game, valid_moves)
            if not (playerOne or playerTwo):
                AI_1_turn = not AI_1_turn
            # If there are no human players, changes AI turns
            game.makeMove(AI_move)
            move_made = True
            animate = animations
            time.sleep(0.1)

        if move_made:
            if animate and game.moves_hist != []:
                animateMove(game.moves_hist[-1], screen, game.board, clock)
            valid_moves = game.get_valid_moves()
            move_made = False

        displayGameState(screen, game, valid_moves, selectedSquare, LogFont)

        if game.checkmate or game.stalemate:
            gameOver = True
            bw = 'Black wins by checkmate'
            ww = 'White wins by checkmate'
            draw = 'Stalemate'
            text = draw if game.stalemate else bw if game.white_move else ww
            drawEndGameText(screen, text)

        clock.tick(MAX_FPS)
        pg.display.flip()

        if gameOver:
            manageGameOver()
            displayGameState(screen, game, valid_moves, selectedSquare,
                             LogFont)

        clock.tick(MAX_FPS)
        pg.display.flip()

    print('End of program.')


def displayGameState(screen, game, valid_moves, selectedSquare, LogFont):
    drawBoard(screen)
    higlightSquares(screen, game, valid_moves, selectedSquare)
    drawPieces(screen, game.board)
    drawMoveLog(screen, game, LogFont)


def drawBoard(screen):
    colors = [pg.Color(COLORS['white']), pg.Color(COLORS['black'])]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r+c) % 2]
            pg.draw.rect(screen, color, pg.Rect(c * SQUARE_SIZE,
                                                r * SQUARE_SIZE,
                                                SQUARE_SIZE,
                                                SQUARE_SIZE))


def higlightSquares(screen, game, valid_moves, selectedSquare):
    if selectedSquare is not None:
        r, c = selectedSquare
        if game.board[r, c][0] == ('w' if game.white_move else 'b'):
            # Higlight selected square
            s = pg.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(100)  # transparency value (0 transparent, 255 opaque)
            s.fill(pg.Color('blue'))
            screen.blit(s, (c * SQUARE_SIZE, r * SQUARE_SIZE))
            # Highlight moves from that square
            s.fill(pg.Color('yellow'))
            for move in valid_moves:
                if move.start_row == r and move.start_col == c:
                    screen.blit(s, (move.end_col * SQUARE_SIZE,
                                    move.end_row * SQUARE_SIZE))


def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != '--':
                screen.blit(IMAGES[piece], pg.Rect(c * SQUARE_SIZE,
                                                   r * SQUARE_SIZE,
                                                   SQUARE_SIZE,
                                                   SQUARE_SIZE))


def drawMoveLog(screen, game, font):
    moveLogRect = pg.Rect(WIDTH, 0, MOVE_LOG_WIDTH, MOVE_LOG_HEIGHT)
    pg.draw.rect(screen, pg.Color('black'), moveLogRect)
    moveLog = game.moves_hist
    moveTexts = [str(m) for m in moveLog]
    padding = MOVE_LOG_WIDTH // 10
    lineSpacing = 2
    for i in range(len(moveTexts)):
        text = ((i % 2) == 0) * (str(1 + i // 2) + '.') + moveTexts[i]
        textObject = font.render(text, True, pg.Color('white'))
        x = padding * (1 + 4 * (i % 2))
        y = padding * (1 + i // 2) + lineSpacing
        textLocation = moveLogRect.move(x, y)
        screen.blit(textObject, textLocation)
    # screen.blit()


def animateMove(move, screen, board, clock):
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    framesPerSquare = 5
    if abs(dR) == abs(dC):
        framesPerSquare = framesPerSquare // 2
        # reduce frames per square in case of diagonal move
        # otherwise they appear significantly slower than other moves
    framesCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(framesCount + 1):
        r, c = (move.start_row + dR * frame / framesCount,
                move.start_col + dC * frame / framesCount)
        drawBoard(screen)
        drawPieces(screen, board)
        # erase the piece moved from its ending square
        color = (move.end_row + move.end_col) % 2
        color = COLORS[list(COLORS.keys())[color]]
        endSquare = pg.Rect(move.end_col * SQUARE_SIZE,
                            move.end_row * SQUARE_SIZE,
                            SQUARE_SIZE,
                            SQUARE_SIZE)
        pg.draw.rect(screen, color, endSquare)
        # draw captured piece into rectangle
        if move.piece_captured != '--':
            if move.is_en_passant:
                enPassantRow = (move.end_row + 1) if\
                    move.piece_moved[0] == 'w' else move.end_row - 1
                # For en passant move, let the captured pawn on its row and
                # instead of displaying it on the enpassant square
                endSquare = pg.Rect(move.end_col * SQUARE_SIZE,
                                    enPassantRow * SQUARE_SIZE,
                                    SQUARE_SIZE,
                                    SQUARE_SIZE)
            screen.blit(IMAGES[move.piece_captured], endSquare)
        # draw moving piece
        if move.piece_moved != '--':
            screen.blit(IMAGES[move.piece_moved], pg.Rect(c * SQUARE_SIZE,
                                                          r * SQUARE_SIZE,
                                                          SQUARE_SIZE,
                                                          SQUARE_SIZE))
        pg.display.flip()
        clock.tick(60)


def drawEndGameText(screen, text):
    font = pg.font.SysFont('TimesNewRoman', 32, True, False)
    textObject = font.render(text, 0, pg.Color('black'))
    textLocation = pg.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 -
                                                     textObject.get_width()
                                                     / 2, HEIGHT / 2 -
                                                     textObject.get_height()
                                                     / 2)
    screen.blit(textObject, textLocation)


if __name__ == '__main__':
    main(playerOne=True, playerTwo=True, algo1='negamax', algo2=None,
         animations=True)
