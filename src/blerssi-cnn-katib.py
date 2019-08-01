from __future__ import absolute_import, division, print_function, unicode_literals
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import argparse
import json
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import os
import pdb
import math
tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

FLAGS = None
BLE_RSSI = pd.read_csv('/opt/iBeacon_RSSI_Labeled.csv') 

path='/opt/iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)


TF_MODEL_DIR = os.getenv("TF_MODEL_DIR", "/train/")
TF_EXPORT_DIR = os.getenv("TF_EXPORT_DIR", "/train/")
TF_MODEL_TYPE = os.getenv("TF_MODEL_TYPE", "DNN")
TF_TRAIN_STEPS = int(os.getenv("TF_TRAIN_STEPS", 10000))

def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x


path='/opt/iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)
x['x'] = x['location'].str[0]
x['y'] = x['location'].str[1:]
x.drop(["location"], axis = 1, inplace = True)
x["x"] = x["x"].apply(fix_pos)
x["y"] = x["y"].astype(int)
y = x.iloc[:, -2:]
x = x.iloc[:, 1:-2]


img_x = np.zeros(shape = (x.shape[0], 25, 25, 1, ))
beacon_coords = {"b3001": (5, 9),
                 "b3002": (9, 14),
                 "b3003": (13, 14),
                 "b3004": (18, 14),
                 "b3005": (9, 11),
                 "b3006": (13, 11),
                 "b3007": (18, 11),
                 "b3008": (9, 8),
                 "b3009": (2, 3),
                 "b3010": (9, 3),
                 "b3011": (13, 3),
                 "b3012": (18, 3),
                 "b3013": (22, 3),}
for key, value in beacon_coords.items():
    img_x[:, value[0], value[1], 0] -= x[key].values/200
    print(key, value)







COLUMNS = list(BLE_RSSI.columns)
FEATURES = COLUMNS[2:]
LABEL = [COLUMNS[0]]

def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x

def conv_block(input_layer, mode, filters=64, dropout=0.0):
    conv = tf.layers.conv2d(inputs=input_layer,
                            filters=filters,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    return pool

def network(feature_input, labels, mode):
    """
    Creates a simple multi-layer convolutional neural network

    :param feature_input:
    :param labels:
    :param mode:
    :return:
    """


    filters = [3, 6, 12]
    dropout_rates = [0.2, 0.3, 0.4]

    conv_layer = tf.reshape(feature_input, [-1, 25,25,1]) 

    for filter_num, dropout_rate in zip(filters, dropout_rates):
        conv_layer = conv_block(conv_layer, mode, filters=filter_num, dropout=dropout_rate)

  
    pool4_flat = tf.layers.flatten(conv_layer)

    dense = tf.layers.dense(inputs=pool4_flat, units=2, activation=tf.nn.relu)

    return dense

def model_fn(features, labels, mode,params):
    """
    Creates model_fn for Tensorflow estimator. This function takes features and input, and
    is responsible for the creation and processing of the Tensorflow graph for training, prediction and evaluation.

    Expected feature: {'image': image tensor }

    :param features: dictionary of input features
    :param labels: dictionary of ground truth labels
    :param mode: graph mode
    :param params: params to configure model
    :return: Estimator spec dependent on mode
    """


    print(features)
    print('*****************')
    learning_rate = params['learning_rate']
    beta1 = params['beta1']
    beta2 = params['beta2']

    image_input = features
    cnn_nw = network(image_input, labels, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return get_prediction_spec(cnn_nw)

    joint_loss = get_loss(cnn_nw, labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return get_training_spec(learning_rate, joint_loss, beta1, beta2)

    else:
        return get_eval_spec(cnn_nw, labels, joint_loss)

def get_prediction_spec(cnn_nw):
    """
    Creates estimator spec for prediction

    :param age_logits: logits of age task
    :param logits: logits of gender task
    :return: Estimator spec
    """
    predictions = {
        "xy_val": cnn_nw
    }

    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

def get_loss(cnn_nw, labels):
    """
    Creates joint loss function

    :param age_logits: logits of age
    :param gender_logits: logits of gender task
    :param labels: ground-truth labels of age and gender
    :return: joint loss of age and gender
    """
    xy_loss = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=cnn_nw))

    return xy_loss

def metric_fn(labels, predict):
    return tf.sqrt(tf.losses.mean_squared_error(labels=tf.cast(labels,tf.float64), predictions=predict))

def categorical_accuracy(labels, predictions, weights=None,
              metrics_collections=None,
              updates_collections=None,
              name=None):
    total = tf.cast(tf.math.equal(tf.math.argmax(predictions,axis=-1), tf.math.argmax(labels,axis=-1)),tf.float64)


    categorical_acc, update_categorical_acc = tf.metrics.mean(total)

    if metrics_collections:
        ops.add_to_collections(metrics_collections, categorical_acc)

    if updates_collections:
        ops.add_to_collections(updates_collections, update_categorical_acc)

    return categorical_acc, update_categorical_acc

def get_eval_spec(cnn_nw, labels, loss):
    """
    Creates eval spec for tensorflow estimator
    :param gender_logits: logits of gender task
    :param age_logits: logits of age task
    :param labels: ground truth labels for age and gender
    :param loss: loss op
    :return: Eval estimator spec
    """
    eval_metric_ops = {
        "rmse_loss": tf.metrics.root_mean_squared_error(labels=tf.cast(labels,tf.float64), predictions=cnn_nw),
        "accuracy":  categorical_accuracy(labels=labels, predictions=cnn_nw)
    }


    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops)


def get_training_spec(learning_rate, joint_loss, beta1, beta2):
    """
    Creates training estimator spec

    :param learning rate for optimizer
    :param joint_loss: loss op
    :return: Training estimator spec
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    xy_train_op = optimizer.minimize(
        loss=joint_loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=joint_loss, train_op=xy_train_op)

def main(unused_args):
  tf.logging.set_verbosity(tf.logging.INFO)

  train_x, val_x, train_y, val_y = train_test_split(x,y, test_size = .2, shuffle = False)

  train_x, val_x, train_y, val_y = train_test_split(img_x, y, test_size = .2, shuffle = False)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = train_x,
      y = np.array(train_y),
      batch_size = FLAGS.batch_size,
      num_epochs = 4000,
      shuffle = False,
      queue_capacity = 300,
      num_threads = 1
    )

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = val_x,
      y = np.array(val_y),
      batch_size = FLAGS.batch_size,
      num_epochs = 1,
      shuffle = False,
      queue_capacity = 300,
      num_threads = 1
    )

  predict_input_fn =  tf.estimator.inputs.numpy_input_fn(
      x = val_x,
      num_epochs = 1,
      shuffle = False,
      queue_capacity = 10000,
      num_threads = 1
    )


  config = tf.estimator.RunConfig(model_dir=FLAGS.log_dir, save_summary_steps=500, save_checkpoints_steps=100)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn, params={
          'learning_rate': FLAGS.learning_rate, 'beta1': FLAGS.beta1, 'beta2': FLAGS.beta2
      },model_dir =FLAGS.log_dir,config=config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=4000
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--tf-model-dir',
                      type=str,
                      default='/train/',
                      help='GCS path or local directory.')
  parser.add_argument('--tf-export-dir',
                      type=str,
                      default='/train/',
                      help='GCS path or local directory to export model')
  parser.add_argument('--tf-model-type',
                      type=str,
                      default='DNN',
                      help='Tensorflow model type for training.')
  parser.add_argument('--tf-train-steps',
                      type=int,
                      default=10000,
                      help='The number of training steps to perform.')
  parser.add_argument('--tf-batch-size',
                      type=int,
                      default=100,
                      help='The number of batch size during training')
  parser.add_argument('--tf-learning-rate',
                      type=float,
                      default=0.01,
                      help='Learning rate for training.')
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--beta1', type=float, default=0.1,
                      help='Initial learning rate')
  parser.add_argument('--beta2', type=float, default=0.5,
                      help='Initial learning rate')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Training batch size')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()

