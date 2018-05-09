from data import CSVLoader
from models import SimpleModel
import tensorflow as tf

session = tf.Session()

s = SimpleModel(session)

session.run(tf.global_variables_initializer())

s.train(CSVLoader("./data/data.csv", 128))

session.close()
