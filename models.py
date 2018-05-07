from typing import List, Tuple, Dict
import tensorflow as tf

class EvaluationPredictor:
    def __init__(self, shape:List[int]) -> None:
        self.num_layers = len(shape)
        self.weights = {}
        self.biases = {}
        self.learning_rate = 0.015
        self.epochs = 1000
        self.batch_size = 64
        self.inputs = tf.placeholder(tf.float32, [None, shape[0]], "inputs")
        self.targets = tf.placeholder(tf.float32, [None, shape[-1]], "targets")

        for i, layer_size in enumerate(shape):
            if i == self.num_layers - 1:
                break
            next_layer_size = shape[i+1]
            self.weights[i] = tf.Variable(
                tf.random_normal([layer_size, next_layer_size]), 
                name=f"weight_{i}"
            )
            self.biases[i] = tf.Variable(
                tf.ones([1, next_layer_size]),
                name=f"bias_{i}"
            )
        
        self.model = self.create_model()
        self.error = tf.reduce_sum(tf.square(self.model - self.targets), name="error")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.trainer = self.optimizer.minimize(self.error)
        # TODO
        # change the following code for the real thing when done debugging
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def create_model(self) -> tf.Tensor:
        current = self.inputs
        for i in range(self.num_layers - 1):
            current = tf.sigmoid(
                tf.matmul(current, self.weights[i]) + self.biases[i],
                f"layer_{i}"
            )
        return current
    
    def train(self, x:List[List[float]], y:List[List[float]]) -> None:
        # self.sess.run(tf.global_variables_initializer())
        for i in range(self.epochs):
            # TODO
            # Add batch support
            self.sess.run(self.trainer, feed_dict={
                self.inputs: x,
                self.targets: y
            })
        
    def predict(self, x:List[List[int]]) -> List[List[int]]:
        return self.sess.run(self.model, feed_dict={
            self.inputs:x
        })

if __name__ == "__main__":
    t = EvaluationPredictor([2, 3, 3, 1])
    x = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    def test(l):
        r=[]
        for row in l:
            a = []
            a.append(row[0] and row[1])
            r.append(a)
        return r
    y = test(x)
    print(y)
    t.epochs = 5000
    t.train(x, y)
    print(t.predict([[0, 1]]))
    print(t.predict([[1, 1]]))