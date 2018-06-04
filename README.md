# TensorChessEvaluator
A Tensorflow project that aims to use ai in chess (not involving playing). Currently using deep autoencoder to encode the structures of a chess position; and using it to define similarity between positions.

Integers are used to represent the pieces as follows:

| Piece type | Value|
|------------|------|
|WHITE_PAWN|0b10000001|
|WHITE_ROOK|0b10000010|
|WHITE_KNIGHT|0b10000100|
|WHITE_BISHOP|0b10001000|
|WHITE_QUEEN|0b10010000|
|WHITE_KING|0b10100000|
|BLACK_PAWN|0b01000001|
|BLACK_ROOK|0b01000011|
|BLACK_KNIGHT|0b01000111|
|BLACK_BISHOP|0b01001111|
|BLACK_QUEEN|0b01011111|
|BLACK_KING|0b01100111|

