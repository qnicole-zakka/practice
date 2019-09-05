import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num', 200, 'here is the integer flag')
tf.app.flags.DEFINE_string('string', 'hahaha', 'here is the string flag')

#def main(argv):
 #   print(FLAGS.num, FLAGS.string)

if __name__ == '__main__':
    tf.app.run()
