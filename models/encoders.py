from typing import List
from .base import BaseModel
from data import DataLoader
import tensorflow as tf

class DeepAutoEncoder(BaseModel):
    def __init__(self, sess: tf.Session(), act=tf.nn.sigmoid) -> None:
        self.encoder = None
        self.decoder = None
        self.activation = act
        super().__init__(
            [65, 64, 32, 16, 8, 16, 32, 64, 65],
            sess,
            0.01
        )
        self.decoder_placeholder = tf.placeholder(
            tf.float32,
            [None, self.shape[self.num_layers//2],
            name="decoder_input"]
        )
    
    def build_encoder(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers // 2):
            current = self.activation(
                tf.add(tf.matmul(current, self.weights[i]), self.biases[i]),
                name = "enode_layer"
            )
        self.encoder = current
        return self.encoder
    
    def build_decoder(self) -> tf.Tensor:
        if self.encoder is None:
            raise AssertionError("Encoder can't be `None`. Need to call `build_encoder` before `build_decoder`")
        
        current = self.encoder
        for i in range(self.num_layers//2, self.num_layers-1):
            current = self.activation(
                tf.add(tf.matmul(current, self.weights[i]), self.biases[i]),
                name = "decode_layer"
            )
        self.decoder = current
        return self.decoder

    def build_model(self) -> tf.Tensor:
        self.build_encoder()
        self.build_decoder()
        return self.decoder
    
    def get_trainer(self):
        return (
            tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            .minimize(self.get_loss_function())
        )
    
    def train(self, data_loader: DataLoader, epochs:int=1000, out_freq:int=100):
        for i in range(epochs):
            x, _ = data_loader.next_batch()
            
            feed = {self.inputs:x, self.targets: x}
            self.sess.run(self.trainer, feed_dict=feed)
            if i % out_freq == 0:
                print(self.sess.run(self.loss, feed_dict=feed))
    
    def encode(self, x: List[List[float]]) -> List[List[float]]:
        return self.sess.run(self.encoder, feed_dict={
            self.inputs: x
        })