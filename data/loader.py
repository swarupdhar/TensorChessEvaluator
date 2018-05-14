from typing import Tuple, List
import csv
import numpy as np
import chess

class DataLoader:
    def __init__(self, file_name:str, batch_size:int):
        self.batch_size = batch_size
        self.file_name = file_name
        self.data_entries = self.get_data()
        self.current_index = 0

    def get_data(self):
        raise NotImplementedError
    
    def next_batch(self) -> Tuple[List[List[float]], List[List[float]]]:
        raise NotImplementedError

class CSVLoader(DataLoader):
    def __init__(self, file_name:str, batch_size:int = 32):
        super().__init__(file_name, batch_size)
    
    def get_data(self) -> List[List[str]]:
        result = []
        with open(self.file_name, 'r') as csv_file:
            next(csv_file, None)
            for line in csv_file:
                row_data = []
                for item in line.strip().split(","):
                    row_data.append(float(item.strip()))
                result.append(row_data)
        return result
    
    def next_batch(self) -> Tuple[List[List[float]], List[List[float]]]:
        np.random.shuffle(self.data_entries)
        if self.current_index + self.batch_size >= len(self.data_entries):
            np.random.shuffle(self.data_entries)        
            self.current_index = 0
        next_index = self.current_index + self.batch_size
        data = self.data_entries[self.current_index: next_index]
        self.current_index = next_index
        inputs = []
        targets = []

        for row in data:
            row_input_data = []
            row_target_data = []

            for items in row[:-1]:
                row_input_data.append(items)
            row_target_data.append(row[-1])
            
            inputs.append(row_input_data)
            targets.append(row_target_data)
        return (inputs, targets)

class PgnLoader(DataLoader):
    def __init__(self, file_name:str, engine_path:str, batch_size:int = 32):
        self.engine = chess.uci.popen_engine(engine_path)
        self.info_handler = chess.uci.InfoHandler()
        self.engine.info_handlers.append(self.info_handler)
        self.piece_encoding_dict = {
            chess.WHITE: {
            chess.PAWN:     0b10000001,
            chess.ROOK:     0b10000010,
            chess.KNIGHT:   0b10000100,
            chess.BISHOP:   0b10001000,
            chess.QUEEN:    0b10010000,
            chess.KING:     0b10100000
            },
            chess.BLACK: {
                chess.PAWN:     0b01000001,
                chess.ROOK:     0b01000011,
                chess.KNIGHT:   0b01000111,
                chess.BISHOP:   0b01001111,
                chess.QUEEN:    0b01011111,
                chess.KING:     0b01100111
            }
        }
        super().__init__(file_name, batch_size)
    
    def encode(board:chess.Board,
               evaluation:int,
               piece_encoding:Dict[bool, Dict[int, int]]) -> List[float]:
        piece_map = board.piece_map()
        encoding = []
        for row in range(8):
            for col in range(8):
                square_index = 56 - (row * 8) + col
                
                if square_index in piece_map:
                    piece = piece_map[square_index]
                    encoding.append(
                        piece_encoding[piece.color][piece.piece_type]
                    )
                else:
                    encoding.append(0)
        if board.turn == chess.WHITE:
            encoding.append(1)
        else:
            encoding.append(0)
        encoding.append((evaluation/100))
        return encoding

    def get_score(handler):
        with handler:
            if 1 in handler.info["score"]:
                return handler.info["score"][1].cp
            else:
                return None

    def get_data(self) -> List[List[str]]:
        result = []
        engine.uci()
        engine.setoption({"UCI_Chess960": True})
        with open(self.file_name, mode='r') as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            while game is not None:
                board = game.board()
                engine.ucinewgame()
                for move in game.main_line():
                    board.push(move)
                    engine.position(board)
                    engine.go(depth=engine_depth)
                    score = self.get_score(info_handler)
                    num_tries = 0
                    while score is None and num_tries < 3:
                        num_tries += 1
                        time.sleep(sleep_time)
                        score = self.get_score(info_handler)
                    
                    if score is None:
                        engine.stop()
                        continue
                    
                    if board.turn == chess.BLACK:
                        score *= -1
                    result.append(self.encode(board, score, self.piece_encoding))
                
                game = chess.pgn.read_game(pgn_file)
        return result
    
    def next_batch(self) -> Tuple[List[List[float]], List[List[float]]]:
        np.random.shuffle(self.data_entries)
        if self.current_index + self.batch_size >= len(self.data_entries):
            np.random.shuffle(self.data_entries)        
            self.current_index = 0
        next_index = self.current_index + self.batch_size
        data = self.data_entries[self.current_index: next_index]
        self.current_index = next_index
        inputs = []
        targets = []

        for row in data:
            row_input_data = []
            row_target_data = []

            for items in row[:-1]:
                row_input_data.append(items)
            row_target_data.append(row[-1])
            
            inputs.append(row_input_data)
            targets.append(row_target_data)
        return (inputs, targets)