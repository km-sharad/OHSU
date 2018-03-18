import numpy as np
import tensorflow as tf
import random
import vgg16_cat_and_dog_svm
import utils_svm
import os
import time
from sklearn import svm
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

train_dir = "/g/kumarsha/project/data/training/"
test_dir = "/g/kumarsha/project/data/testing/"
batch_size = 25 

# conf = SparkConf().setAppName("Spark_SVM")
# sc = SparkContext(conf=conf)

def getTrainBatchImages(cat_image_names, dog_image_names):
    image_names = []
    for i in xrange(len(cat_image_names)):
        image_names.append(str(train_dir) + 'cat/' + str(cat_image_names[i]))
        image_names.append(str(train_dir) + 'dog/' + str(dog_image_names[i]))

    return image_names

def sparkRun(sess, imagePath):
    # images = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])
    images = tf.placeholder("float", [1,224, 224, 3])

    vgg = vgg16_cat_and_dog_svm.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    sess.run(tf.global_variables_initializer())

    relu7 = vgg.relu7

    image = utils_svm.load_image(imagePath)
    image = np.reshape(image, (1,224,224,3))

    train_activations = sess.run(relu7, feed_dict={images: image})
    return train_activations

def activationRun(image_filenames):
    print('########image_filename act: ', image_filenames)

    with tf.Graph().as_default() as g:        
        with tf.Session() as sess:     
            start_time = time.time()
            activations = sparkRun(sess, image_filenames)
            return activations

start_time = time.time()

cat_image_names_train = os.listdir(str(train_dir) + 'cat/')
dog_image_names_train = os.listdir(str(train_dir) + 'dog/')
random.shuffle(cat_image_names_train)
random.shuffle(dog_image_names_train)   

image_filenames = getTrainBatchImages(cat_image_names_train[0:25], dog_image_names_train[0:25])  

imageNames = sc.parallelize(image_filenames)
sc.addPyFile("vgg16_cat_and_dog_svm.py")
sc.addPyFile("utils_svm.py")

train_activations = imageNames.map(activationRun)
train_activations_collect = train_activations.collect()
print('train_activations_len: ', len(train_activations_collect))
print('train_activations_collect: ', train_activations_collect[0].shape)
print('duration: ' + str((time.time() - start_time)))
