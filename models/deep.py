from typing import List
from .base import BaseModel
import tensorflow as tf

class LinearOutputLayerNetwork(BaseModel):
    def __init__(self, sess:tf.Session()) -> None:
        super().__init__(
            [65, 65, 65, 64, 64, 32, 32, 16, 16, 16, 1],
            sess,
            0.01
        )
    
    def get_trainer(self):
        return (
            tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                .minimize(self.get_loss_function())
        )
    
    def build_model(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers - 1):
            if i >= self.num_layers - 2:
                current = tf.add(
                    tf.matmul(current, self.weights[i]),
                    self.biases[i],
                    name="layer_out"
                )
            else:
                current = tf.sigmoid(
                    tf.matmul(current, self.weights[i]) + self.biases[i],
                    name=f"layer{i}"
                )
        return current