import numpy as np
import tensorflow as tf
import random
import vgg16_cat_and_dog
import utils
import os

data_dir = "../../data/training/"

total_training_images = 7500    # Number of training images where car door handle is visible
total_validation_images = 2500 # Number of validation images where car door handle is visible
max_epochs = 500                        # Number of batches to run
batch_size = 10 

def getBatchImages(cat_image_names, dog_image_names):
    images = []
    gt_1_hot_vec = []
    for i in xrange(batch_size):
        images.append(utils.load_image(str(data_dir) + 'cat/' + str(cat_image_names[i])))
        gt_1_hot_vec.append([1,0])
        images.append(utils.load_image(str(data_dir) + 'dog/' + str(dog_image_names[i])))
        gt_1_hot_vec.append([0,1])

    return images, gt_1_hot_vec

# saver = tf.train.Saver(max_to_keep=10)

with tf.device('/cpu:0'):
    with tf.Session() as sess:        
        # Restore variables from disk.
        # saver.restore(sess, "./ckpt/model4845.ckpt")
        # print("Model restored.")

        images = tf.placeholder("float", [20, 224, 224, 3])
        gt = tf.placeholder(dtype=tf.float32, shape=[batch_size*2,2])
        # feed_dict = {images: batch_images}

        vgg = vgg16_cat_and_dog.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images, gt)
        sess.run(tf.global_variables_initializer())

        prob = vgg.prob

        train_step = vgg.train(prob, gt)

        cat_image_names = os.listdir(str(data_dir) + 'cat/')
        dog_image_names = os.listdir(str(data_dir) + 'dog/')

        for epoch in xrange(max_epochs):
            random.shuffle(cat_image_names)
            random.shuffle(dog_image_names)

            for batch in xrange(len(dog_image_names)/batch_size):
                batch_images, gt_1_hot_vec = \
                    getBatchImages(cat_image_names[batch * batch_size : (batch * batch_size) + batch_size], \
                                dog_image_names[batch * batch_size : (batch * batch_size) + batch_size])

                var_list = sess.run(train_step, feed_dict={images: batch_images, gt: gt_1_hot_vec})
                print('batch: ', batch)
                # print('var_list: ', var_list[0].shape, var_list[1].shape, var_list[2].shape, var_list[3].shape)


            # Save the variables to disk.
            # ckpt_file = './ckpt/model' + str(epoch) + '.ckpt'
            # save_path = saver.save(sess, ckpt_file)
            # print("Model saved in file: %s" % save_path)                

