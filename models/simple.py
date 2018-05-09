from .base import BaseModel
import tensorflow as tf

class SimpleModel(BaseModel):
    def __init__(self, sess:tf.Session):
        super().__init__([65, 3, 1], sess)
    
    def build_model(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers - 1):
            current = tf.sigmoid(
                tf.matmul(current, self.weights[i]) + self.biases[i],
                f"layer_{i}"
            )
        return current
