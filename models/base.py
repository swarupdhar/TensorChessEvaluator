from typing import List

from data import DataLoader
import tensorflow as tf

class BaseModel:
    def __init__(self, config) -> None:
        if "shape" not in config:
            raise ValueError("Config must have a shape entry of type List describing the shape of the network")
        self.num_layers = len(config["shape"])
        self.shape = config["shape"]
        self.learning_rate = config["lr"] if "lr" in config else 0.01
        self.epochs = config["epochs"] if "epochs" in config else 1000
        self.batch_size = config["batch_size"] if "batch_size" in config else 64
        
        self.weights = {}
        self.biases = {}
        
        self.inputs = tf.placeholder(tf.float32, [None, self.shape[0]], "inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.shape[-1]], "targets")
        
        self.init_weights_and_biases()
        self.model = self.build_model()

        self.sess = config["sess"] if "sess" in config else tf.Session()
        self.loss = self.get_loss_function()
        self.trainer = self.get_trainer()
    
    def init_weights_and_biases(self):
        raise NotImplementedError
    
    def build_model(self) -> tf.Tensor:
        raise NotImplementedError
    
    def get_loss_function(self):
        return tf.reduce_sum(tf.square(self.model - self.targets))

    def get_trainer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.get_loss_function())

    def train(self, data_loader:DataLoader):
        for i in range(self.epochs):
            x, y = data_loader.next_batch()
            self.sess.run(self.trainer, feed_dict={
                self.inputs: x,
                self.targets: y
            })
    
    def predict(self, x:List[List[float]]) -> List[List[float]]:
        return self.sess.run(self.model, feed_dict={
            self.inputs: x
        })
