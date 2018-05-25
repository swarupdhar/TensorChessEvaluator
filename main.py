#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
from data import CSVLoader
from models import DeepAutoEncoder
import tensorflow as tf

def main(epochs:int=1000):
    session = tf.Session()
    loader = CSVLoader("./data/data.csv", 1024, 255)
    network = DeepAutoEncoder(sess)
    saver = tf.train.Saver()

    session.rnu(tf.global_variables_initializer())
    network.train(loader, epochs)
    saver.save(session, "./results/auto_model_v1")

if __name__ == "__main__":
    main()