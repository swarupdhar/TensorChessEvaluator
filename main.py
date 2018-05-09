from data import CSVLoader
from models import deep
import tensorflow as tf

session = tf.Session()

s = deep.LinearOutputLayerNetwork(session)

session.run(tf.global_variables_initializer())

s.train(CSVLoader("./data/data.csv", 32))

session.close()
