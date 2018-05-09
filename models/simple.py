from .base import BaseModel
import tensorflow as tf

class SimpleModel(BaseModel):
    def __init__(self, sess:tf.Session):
        super().__init__({
            'shape' : [65, 3, 1],
            'sess': sess
        })
    
    def init_weights_and_biases(self):
        for i, layer_size in enumerate(self.shape):
            if i == self.num_layers - 1:
                break
            next_layer_size = self.shape[i+1]
            self.weights[i] = tf.Variable(
                tf.random_normal([layer_size, next_layer_size]), 
                name=f"weight_{i}"
            )
            self.biases[i] = tf.Variable(
                tf.ones([1, next_layer_size]),
                name=f"bias_{i}"
            )
    
    def build_model(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers - 1):
            current = tf.sigmoid(
                tf.matmul(current, self.weights[i]) + self.biases[i],
                f"layer_{i}"
            )
        return current
