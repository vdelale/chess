# This script contains all the information about the current state of the board.
# It also enables to know every possible move for the current board.

import numpy as np
from copy import deepcopy
from itertools import product
class Game():
    def __init__(self):
        # The board is represented by a 8 x 8 array. Each
        # cell contains 2 characters, the first being the color
        # of the piece, and the second one the type of the piece.
        # '--' stands for an empty cell
        pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        w_pieces = ['w' + piece for piece in pieces]
        b_pieces = ['b' + piece for piece in pieces]
        empty = ['--'] * 8
        w_pawns = ['wp'] * 8
        b_pawns = ['bp'] * 8
        self.board = np.array([b_pieces,
                               b_pawns,
                               empty,
                               empty,
                               empty,
                               empty,
                               w_pawns,
                               w_pieces])
        self.white_move = True
        self.moves_hist = []
        self.moveFunctions = {'p': self.pawn_moves, 'r': self.rook_moves,
                              'n': self.knight_moves, 'b': self.bishop_moves,
                              'q': self.queen_moves, 'k': self.king_moves}
        self.wk_location = [7, 4]
        self.bk_location = [0, 4]
        self.in_check = False
        self.pins = []
        self.checks = []
        self.checkmate = False
        self.stalemate = False
        self.enpassant = None # will be the coordinates for the en passant cell
        self.enpassantLog = [self.enpassant]
        self.currentCastleRights = CastleRights(True, True, True, True)
        self.castleRightsLog = [CastleRights(self.currentCastleRights.wks,
                                             self.currentCastleRights.wqs,
                                             self.currentCastleRights.bks,
                                             self.currentCastleRights.bqs)]
                            
    def makeMove(self, move):
        self.board[move.start_row][move.start_col] = '--'
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.white_move = not self.white_move
        if move.piece_moved == 'wk':
            self.wk_location = [move.end_row, move.end_col]
        elif move.piece_moved == 'bk':
            self.bk_location = [move.end_row, move.end_col]
        
        # Pawn promotion, en passant and castling cannot happen at the same time
        # therefore we can use elif conditions to speed up the process
        if move.pawn_promotion:
            promotion = ''
            while promotion not in ['q', 'r', 'b', 'n']:
                promotion = 'Q' #input('Promote to Q, R, B or N:')
                promotion = promotion.lower()
            promotion = move.piece_moved[0] + promotion
            self.board[move.end_row][move.end_col] = promotion

        elif move.is_en_passant:
            self.board[move.start_row][move.end_col] = "--"
        
        elif move.isCastleMove:
            if move.end_col - move.start_col > 0: # kingside castle
                self.board[move.end_row, move.end_col - 1] = \
                    self.board[move.end_row, move.end_col + 1] # moves the rook
                self.board[move.end_row, move.end_col + 1] = '--' # erases rook
            else: # queenside castle
                self.board[move.end_row, move.end_col + 1] = \
                    self.board[move.end_row, move.end_col - 2] # moves the rook
                self.board[move.end_row, move.end_col - 2] = '--' # erases rook

        # Update en passant possible
        two_cell_move = abs(move.start_row - move.end_row) == 2
        if move.piece_moved[1] == 'p' and two_cell_move:
            enpassant_row = (move.start_row + move.end_row) // 2
            # deals with both color cases
            self.enpassant = (enpassant_row, move.start_col)
        else:
            self.enpassant = None
        self.enpassantLog.append(self.enpassant)
        # Update castling rights
        self.update_castle_rights(move)
        self.castleRightsLog.append(CastleRights(self.currentCastleRights.wks,
                                                 self.currentCastleRights.wqs,
                                                 self.currentCastleRights.bks,
                                                 self.currentCastleRights.bqs))
        game_bis = deepcopy(self)
        game_bis.get_valid_moves()
        move.checks = len(game_bis.checks)
        move.checkmates = game_bis.checkmate
        del(game_bis)        
        self.moves_hist.append(move)                                           


    def undoMove(self):
        if len(self.moves_hist) > 0:
            move = self.moves_hist.pop()
            self.board[move.start_row][move.start_col] = move.piece_moved
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.white_move = not self.white_move
            if move.piece_moved == 'wk':
                self.wk_location = [move.start_row, move.start_col]
            elif move.piece_moved == 'bk':
                self.bk_location = [move.start_row, move.start_col]
            # undo en passant
            if move.is_en_passant:
                self.board[move.end_row][move.end_col] = '--'
                self.board[move.start_row][move.end_col] = move.piece_captured
                self.enpassant = (move.end_row, move.end_col)
            elif move.isCastleMove:
                if move.end_col - move.start_col > 0: # kingside castle
                    self.board[move.end_row, move.end_col + 1] = \
                        self.board[move.end_row, move.end_col - 1] 
                    self.board[move.end_row, move.end_col - 1] = '--' 
                else: # queenside castle
                    self.board[move.end_row, move.end_col - 2] = \
                        self.board[move.end_row, move.end_col + 1]
                    self.board[move.end_row, move.end_col + 1] = '--' 
            # undo a 2 cell pawns advance
            two_cell_move = abs(move.start_row - move.end_row) == 2
            if move.piece_moved[1] == 'p' and two_cell_move:
                self.enpassant = None
            # reset the enpassant cell to the previous value after undoing move
            self.enpassantLog.pop()
            self.enpassant = self.enpassantLog[-1]
            # undo castling rights
            self.castleRightsLog.pop() # get rid of the new castling rights
            self.currentCastleRights = deepcopy(self.castleRightsLog[-1])
            # set current castle rights to the last on in the list
            self.checkmate = False
            self.stalemate = False

    def update_castle_rights(self, move):
        if move.piece_moved == 'wk':
            self.currentCastleRights.wks = False
            self.currentCastleRights.wqs = False
        elif move.piece_moved == 'bk':
            self.currentCastleRights.bks = False
            self.currentCastleRights.bqs = False
        elif move.piece_moved == 'wr':
            if move.start_row == 7:
                if move.start_col == 0:
                    self.currentCastleRights.wqs = False
                elif move.start_col == 7:
                    self.currentCastleRights.wks = False
        elif move.piece_moved == 'br':
            if move.start_row == 0:
                if move.start_col == 0:
                    self.currentCastleRights.bqs = False
                elif move.start_col == 7:
                    self.currentCastleRights.bks = False
        if move.piece_captured[1] == 'r':
            if move.end_row == 0:
                if move.end_col == 0:
                    self.currentCastleRights.bqs = False
                elif move.end_col ==7:
                    self.currentCastleRights.bks = False
            elif move.end_row == 7:
                if move.end_col == 0:
                    self.currentCastleRights.wqs = False
                elif move.end_col ==7:
                    self.currentCastleRights.wks = False

    def pawn_moves(self, r, c, moves):
        dir =  1 - 2 * self.white_move  # direction: -1 for white 1 for black
        turn = 'w' if self.white_move else 'b'
        opponent = 'b' if self.white_move else 'w'
        # Opposite direction of the movement depending on the color
        start_row = int((5 / 2) * (1 - dir)) + 1 # 6 for white 1 for black
        last_row = int((7 / 2) * (1 + dir)) # 0 for white 7 for black
        
        # Verify whether piece is pinned
        is_pinned = False
        pin_direction = None
        for i in range(len(self.pins) -1, -1 , -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                is_pinned = True
                pin_direction = self.pins[i][2], self.pins[i][3]
                self.pins.remove(self.pins[i])
                break
            
        pawn_promotion = (r + dir) == last_row
        # Move without captures
        if self.board[r+dir][c] == '--':
            if (not is_pinned) or pin_direction in [(-1, 0), (1, 0)]:
                moves.append(Move((r, c), (r + dir, c), self,
                                  pawn_promotion=pawn_promotion))
                if r == start_row and self.board[start_row + 2 * dir][c] == '--':
                    moves.append(Move((r, c), (r + 2 * dir, c), self))
        # Capture on the left
        if ((c - 1) >= 0):
            enpassant = (r + dir, c - 1) == self.enpassant
            opponent_on_end = self.board[r + dir][c - 1][0] == opponent
            if opponent_on_end:
                if (not is_pinned) or pin_direction == (dir, -1):
                    moves.append(Move((r, c), (r + dir, c - 1), self,
                                      pawn_promotion=pawn_promotion))
            # we can use elif since both cases cannot happen simultaneously
            elif enpassant: 
                # We need to check the special case where the en passant 
                # clears two pawns on the rank, and makes check possible
                # by a rook or a queen if a king is on the rank
                kingRow, kingCol = self.wk_location if self.white_move else\
                     self.bk_location
                attackingPiece, blockingPiece = False, False
                if kingRow == r:
                    if kingCol < c:  # king on the left of the pawn
                        # inside range between king and pawn
                        # outside, between pawn and border
                        insideRange = range(kingCol + 1, c - 1)
                        outsideRange = range(c + 1, 8)
                    else:  # king on the right of the pawn
                        insideRange = range(kingCol - 1, c, -1)
                        outsideRange = range(c - 2, - 1, -1)
                    for i in insideRange:
                        if self.board[r][i] != '--':
                            blockingPiece = True
                            break
                    for i in outsideRange:
                        square = self.board[r][i]
                        if square[0] == opponent and square[1] in ('q', 'r'):
                            attackingPiece = True
                        elif square != '--':
                            blockingPiece = True
                if not attackingPiece or blockingPiece:
                    moves.append(Move((r, c), (r + dir, c - 1), self,
                                      en_passant=True))

        # Capture on the right
        if ((c + 1) < 8):
            enpassant = (r + dir, c + 1) == self.enpassant
            opponent_on_end = self.board[r + dir][c + 1][0] == opponent
            if opponent_on_end:
                if (not is_pinned) or pin_direction == (dir, 1):
                    moves.append(Move((r, c), (r + dir, c + 1), self,
                                      pawn_promotion=pawn_promotion))
            elif enpassant: 
                # Only slightly different from above, check for comments
                kingRow, kingCol = self.wk_location if self.white_move else\
                     self.bk_location
                attackingPiece, blockingPiece = False, False
                if kingRow == r:
                    if kingCol < c:
                        insideRange = range(kingCol + 1, c )
                        outsideRange = range(c + 2, 8)
                    else:
                        insideRange = range(kingCol - 1, c + 1, -1)
                        outsideRange = range(c - 1, - 1, -1)
                    for i in insideRange:
                        if self.board[r][i] != '--':
                            blockingPiece = True
                            break
                    for i in outsideRange:
                        square = self.board[r][i]
                        if square[0] == opponent and square[1] in ('q', 'r'):
                            attackingPiece = True
                        elif square != '--':
                            blockingPiece = True
                if not attackingPiece or blockingPiece:
                    moves.append(Move((r, c), (r + dir, c + 1), self,
                                      en_passant=True)) 

    def rook_moves(self, r, c, moves):
        turn = 'w' * self.white_move + 'b' * (1 - self.white_move)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        is_pinned = False
        pin_direction = None
        for i in range(len(self.pins) -1, -1 , -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                is_pinned = True
                pin_direction = self.pins[i][2], self.pins[i][3]
                if self.board[r, c][1] != 'q':
                    # Don't remove queen from pin on rook moves
                    # Only remove it on bishop moves
                    self.pins.remove(self.pins[i])
                break
        
        for d1, d2 in directions:
            i, j = d1, d2
            in_board = (0 <= (r + i) < 8) and (0 <= (c +j) < 8)
            while in_board and self.board[r + i, c + j] == '--':
                if (not is_pinned) or pin_direction in [(d1, d2), (-d1, -d2)]:
                    moves.append(Move((r, c), (r + i, c + j), self))
                i += d1
                j += d2
                in_board = (0 <= (r + i) < 8) and (0 <= (c +j) < 8)
            if in_board and self.board[r + i, c + j][0] != turn:
                if (not is_pinned) or pin_direction in [(d1, d2), (-d1, -d2)]:
                    moves.append(Move((r, c), (r + i, c + j), self))

    def knight_moves(self, r, c, moves):
        turn = 'w' * self.white_move + 'b' * (1 - self.white_move)
        directions = [-2, -1, 1, 2]
        directions = product(directions, directions)
        directions = [(i, j) for i, j in directions if abs(i) != abs(j)]
        # Only eight possible moves for a knight at most 

        is_pinned = False
        for i in range(len(self.pins) -1, -1 , -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                is_pinned = True
                self.pins.remove(self.pins[i])
                break

        for i, j in directions:
            in_board = (0 <= (r + i) < 8) and (0<= (c + j) < 8)
            if in_board and self.board[r + i, c +j][0] != turn:
                if not is_pinned:
                    moves.append(Move((r, c), (r +i, c + j), self))

    def bishop_moves(self, r, c, moves):
        turn = 'w' * self.white_move + 'b' * (1 - self.white_move)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        is_pinned = False
        pin_direction = None
        for i in range(len(self.pins) -1, -1 , -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                is_pinned = True
                pin_direction = self.pins[i][2], self.pins[i][3]
                self.pins.remove(self.pins[i])
                break

        for d1, d2 in directions:
            i, j = d1, d2
            in_board = (0 <= (r + i) < 8) and (0 <= (c + j) < 8)
            while in_board and self.board[r + i, c+j] =='--':
                move_pin_dir = pin_direction in [(d1, d2), (-d1, -d2)]
                if (not is_pinned) or move_pin_dir:
                    moves.append(Move((r, c), (r + i, c + j), self))
                i += d1
                j += d2
                in_board = (0 <= (r + i) < 8) and (0 <= (c + j) < 8)
            if in_board and self.board[r + i, c + j][0] != turn:
                move_pin_dir = pin_direction in [(d1, d2), (-d1, -d2)]
                if (not is_pinned) or move_pin_dir:
                    moves.append(Move((r, c), (r + i, c + j), self))


    def queen_moves(self, r, c, moves):
        self.rook_moves(r, c, moves)
        self.bishop_moves(r, c, moves)
        

    def king_moves(self, r, c, moves):
        turn = 'w' * self.white_move + 'b' * (1 - self.white_move)
        directions = [-1, 0, 1]
        directions = list(product(directions, directions))
        directions.remove((0,0))
        # Also eight possible moves for the king
        for i, j in directions:
            in_board = (0 <= (r + i) < 8) and (0<= (c + j) < 8)
            if in_board and (self.board[r + i, c + j][0] != turn):
                # Simulate the move and check whether the king is checked
                temp1 = self.board[r + i, c + j]
                temp2 = self.board[r, c]
                if turn == 'w':
                    self.wk_location = [r + i, c + j]
                    self.board[r + i, c + j] = 'wk'
                    self.board[r, c] = '--'
                else:
                    self.bk_location = [r + i, c + j]
                    self.board[r + i, c + j] = 'bk'
                    self.board[r, c] = '--'
                # Check for checks in new_positions
                in_check, pins, checks = self.check_pins_and_checks()
                # Bring back the king on the right cell
                if turn == 'w':
                    self.wk_location = [r, c]
                    self.board[r, c] = temp2
                    self.board[r + i, c + j] = temp1
                else:
                    self.bk_location = [r, c]
                    self.board[r, c] = temp2
                    self.board[r + i, c + j] = temp1
                if not in_check:
                    moves.append(Move((r, c), (r+i, c+j), self))

    def castle_moves(self, r, c, moves):
        if self.check_pins_and_checks()[0]:
            return None # Can't castle if checked
        if (self.white_move and self.currentCastleRights.wks) or \
           (not self.white_move and self.currentCastleRights.bks):
           self.kingsideCastleMoves(r, c, moves)
        if (self.white_move and self.currentCastleRights.wqs) or \
           (not self.white_move and self.currentCastleRights.bqs):
           self.queensideCastleMoves(r, c, moves)

    def kingsideCastleMoves(self, r, c , moves):
        if self.board[r, c + 1] == '--' and self.board[r, c + 2] == '--' and \
           not self.squareUnderAttack(r, c + 1) and not\
           self.squareUnderAttack(r, c + 2):
            moves.append(Move((r, c), (r, c + 2), self, CastleMove=True))
            
    def queensideCastleMoves(self, r, c , moves):
        if self.board[r, c - 1] == '--' and self.board[r, c - 2] == '--' and\
           self.board[r, c - 3] == '--' and not\
           self.squareUnderAttack(r, c - 1) and not\
           self.squareUnderAttack(r, c - 2):
            moves.append(Move((r, c), (r, c - 2), self, CastleMove=True))

    def get_all_moves(self):
        moves = []
        turn = 'w' * self.white_move + 'b' * (1 - self.white_move)
        for r in range(len(self.board)):
            for c in range(len(self.board)):
                if turn == self.board[r, c][0]:
                    piece = self.board[r, c][1]
                    self.moveFunctions[piece](r, c, moves)
        return moves

    def get_valid_moves(self):
        moves = []
        self.in_check, self.pins, self.checks = self.check_pins_and_checks()
        if self.white_move:
            king_row = self.wk_location[0]
            king_col = self.wk_location[1]
        else:
            king_row = self.bk_location[0]
            king_col = self.bk_location[1]
        # First we generate all the moves and then we will keep only the ones 
        # that don't let the current player's king being checked.
        # We will only deal with move that capture the piece of block the check
        # for now, and later we will deal with moves to escape the check
        if self.in_check:
            if len(self.checks) == 1:
                moves = self.get_all_moves()
                check = self.checks[0]
                check_row = check[0]
                check_col = check[1]
                d = check[2], check[3] # direction of the check
                check_piece = self.board[check_row][check_col]
                valid_squares = []
                if check_piece == 'n':
                    valid_squares = [(check_row, check_col)]
                else:
                    for i in range(1, 8):
                        valid_square = king_row + i * d[0], king_col + i * d[1]
                        valid_squares.append(valid_square)
                        if valid_square == (check_row, check_col):
                            break
                for i in range(len(moves) -1, -1, -1):
                    # Need to iterate the list backwards to remove elements
                    # during iterations
                    if moves[i].piece_moved[1] != 'k':
                        # Isn't a king move
                        end_squares = moves[i].end_row, moves[i].end_col
                        if not end_squares in valid_squares:
                            moves.remove(moves[i])
            else: # double check therefore the king has to move
                self.king_moves(king_row, king_col, moves)
        # else player not in check therefore all move are valid
        # except for pins and moves that put in check
        else:
            moves = self.get_all_moves()
        
        if moves == []:
            if self.in_check:
                self.checkmate = True
            else:
                self.stalemate = True

        if self.white_move:
            self.castle_moves(self.wk_location[0], self.wk_location[1], moves)
        else:
            self.castle_moves(self.bk_location[0], self.bk_location[1], moves)
        return moves

    def check_pins_and_checks(self):
        pins, checks, in_check = [], [], False
        if self.white_move:
            ally_color, enemy_color = 'w', 'b'
            k_row = self.wk_location[0]
            k_col = self.wk_location[1]
        else:
            ally_color, enemy_color = 'b', 'w'
            k_row = self.bk_location[0]
            k_col = self.bk_location[1]
        directions = [-1, 0, 1]
        directions = list(product(directions, directions))
        directions.remove((0,0))
        for j in range(len(directions)):
            d = directions[j]
            possible_pin = None
            i = 1
            while (0 <= k_row + i * d[0] < 8) and (0 <= k_col + i * d[1] < 8):
                end_piece = self.board[k_row + i * d[0], k_col + i * d[1]]
                if end_piece[0] == ally_color:# and end_piece[1] != 'k':
                    # We need to add the condition that the piece must not be
                    # the king because we simulate the move of the king, since we
                    # do not change the board, but only the location of the king
                    # then 
                    if possible_pin is None:
                        possible_pin = (k_row + i * d[0], k_col + i * d[1], *d)
                    else: # 2 allied pieces therefore no pin
                        break
                elif end_piece[0] == enemy_color:
                    type_p = end_piece[1]
                    # Five types of possibles checks here 
                    # (we deal with the knight case outside the loop)
                    ort_check = (d[0] == 0 or d[1]==0) and type_p in ('r', 'q')
                    # orthognal check by rook or queen
                    d_check = abs(d[0]) == abs(d[1]) and type_p in ('b', 'q')
                    # diagonal check by bishop or queen
                    p_check = ally_color == 'w' and d in [(-1, -1), (-1, 1)]
                    p_check |= ally_color == 'b' and d in [(1, -1), (1, 1)]
                    p_check &= type_p == 'p' and i == 1
                    # diagonal check by pawn
                    k_check = i==1 and type_p == 'k'
                    # avoid check by king
                    if ort_check or d_check or p_check or k_check:
                        if possible_pin is None:
                            in_check = True
                            check = (k_row + i*d[0], k_col + i*d[1], d[0], d[1])
                            checks.append(check)
                            break
                        else: # ally piece between therefore pin
                            pins.append(possible_pin)
                            break
                    else: # enemy piece not applying check
                        break
                i += 1

        # Does knight check ?

        directions = [-2, -1, 1, 2]
        directions = product(directions, directions)
        directions = [(i, j) for i, j in directions if abs(i) != abs(j)]
        for m in directions:
            end_row = k_row + m[0]
            end_col = k_col + m[1]
            if (0 <= end_row < 8) and (0 <= end_col < 8):
                end_piece = self.board[end_row, end_col]
                if end_piece[0] == enemy_color and end_piece[1] == 'n':
                    in_check = True
                    checks.append((end_row, end_col, m[0], m[1]))
        return in_check, pins, checks


    def squareUnderAttack(self, r, c):
        self.white_move = not self.white_move # switch to opponent's turn
        opponent_moves = self.get_all_moves()
        self.white_move = not self.white_move # switch back
        for move in opponent_moves:
            if move.end_row == r and move.end_col == c:
                return True
        return False

class CastleRights():
    def __init__(self, wks, wqs, bks, bqs):
        self.wks = wks
        self.wqs = wqs
        self.bks = bks
        self.bqs = bqs



class Move():

    ranks_to_rows = {str(num): 8 - num for num in range(1, 9)}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {chr(num): num - 97 for num in range(97, 105)}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_square, end_square, game, en_passant=False,
                 pawn_promotion=False, CastleMove=False):
        self.game = game
        board = self.game.board
        self.start_row = start_square[0]
        self.start_col = start_square[1]
        self.end_row = end_square[0]
        self.end_col = end_square[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.pawn_promotion = pawn_promotion
        # Dealing with en-passant
        self.is_en_passant = en_passant
        if self.is_en_passant:
            self.piece_captured = 'bp' if game.white_move else 'wp'
        self.isCastleMove = CastleMove
        # We then create an id for every move, that we will use to test equaliy
        # between two moves
        self.move_id = self.start_row * 1000 + self.start_col * 100 
        self.move_id += self.end_row * 10 + self.end_col
        self.checks = 0
        self.checkmates = False
    
    def __str__(self):
        return self.getChessNotation()
    
    def __eq__(self, other):
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False

    def getRankFile(self, r, c):
        return self.cols_to_files[c] + self.rows_to_ranks[r]

    def getChessNotation(self):
        if self.isCastleMove:
            notation = 'O-O' if self.end_col == 6 else 'O-O-O'
        else:
            piece = self.piece_moved[1]
            takes = self.piece_captured != '--'
            start = self.getRankFile(self.start_row, self.start_col)
            end = self.getRankFile(self.end_row, self.end_col)
            ep = ''
            if piece == 'p':
                piece = ''
                if takes:
                    piece = start[0]
                    if self.is_en_passant:
                        ep = ' e.p.'
                else:
                    start = ''
            else:
                piece = piece.upper()
            notation = piece + (1 - len(piece)) * start + 'x' * takes + end
            notation += ep
        if not self.checkmates:
            notation += '+' * self.checks
        else:
            notation += '#'
        return notation
