#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
from data import CSVLoader
from models import LinearOutputLayerNetwork
import tensorflow as tf

session = tf.Session()
loader = CSVLoader("./data/data.csv", 25298)

test_model = LinearOutputLayerNetwork(session)

saver = tf.train.Saver(max_to_keep=2)

session.run(tf.global_variables_initializer())

test_model.train(loader, 100)

x, y = loader.next_batch()

# total_err = 0
# for pred, actual in zip(test_model.predict(x), y):
#     total_err += abs(pred[0] - actual[0])
#     print(f"Actual: {actual}, predicted: {pred}")
# print(total_err/10)


saver.save(session, "./results/model1")
session.close()
