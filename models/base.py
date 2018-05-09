from typing import List, Callable

from data import DataLoader
import tensorflow as tf

class BaseModel:
    def __init__(
        self,
        shape:List[int],
        sess:tf.Session(),
        lr:float=0.01,
        init_method=tf.random_normal) -> None:
        if not shape:
            raise ValueError("Can't have an empty shape")
        self.num_layers = len(shape)
        self.shape = shape
        self.learning_rate = lr
        
        self.weights = {}
        self.biases = {}
        
        self.inputs = tf.placeholder(tf.float32, [None, self.shape[0]], "inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.shape[-1]], "targets")
        
        self.init_weights_and_biases(init_method)
        self.model = self.build_model()

        self.sess = sess
        self.loss = self.get_loss_function()
        self.trainer = self.get_trainer()
    
    def init_weights_and_biases(self, init:Callable):
        for i, layer_size in enumerate(self.shape):
            if i == self.num_layers - 1:
                break
            next_layer_size = self.shape[i+1]
            self.weights[i] = tf.Variable(
                init([layer_size, next_layer_size]), 
                name=f"weight_{i}"
            )
            self.biases[i] = tf.Variable(
                tf.ones([1, next_layer_size]),
                name=f"bias_{i}"
            )
    
    def build_model(self) -> tf.Tensor:
        raise NotImplementedError
    
    def get_loss_function(self):
        return tf.reduce_sum(tf.square(self.model - self.targets))

    def get_trainer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.get_loss_function())

    def train(self, data_loader:DataLoader, epochs=1000):
        for i in range(epochs):
            x, y = data_loader.next_batch()
            self.sess.run(self.trainer, feed_dict={
                self.inputs: x,
                self.targets: y
            })
    
    def predict(self, x:List[List[float]]) -> List[List[float]]:
        return self.sess.run(self.model, feed_dict={
            self.inputs: x
        })
