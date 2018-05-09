from typing import List
from .base import BaseModel
import tensorflow as tf

class LinearOutputLayerNetwork(BaseModel):
    def __init__(self, sess:tf.Session()) -> None:
        super().__init__(
            [65, 64, 64, 32, 32, 32, 16, 16, 16, 4, 1] if not shape else shape,
            sess,
            0.001
        )
        self.sess.run(tf.global_variables_initializer())
    
    def build_model(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers - 1):
            if i >= self.num_layers - 3:
                current = tf.matmul(current, self.weights[i]) + self.biases[i]
            else:
                current = tf.tanh(
                    tf.matmul(current, self.weights[i]) + self.biases[i],
                    f"layer_{i}"
                )
        return current