import numpy as np
import tensorflow as tf
import random
import vgg16_cat_and_dog
import utils
import os

train_dir = "../../data/training/"
test_dir = "../../data/testing/"

total_training_images = 7500    
total_validation_images = 2500 
max_epochs = 50                        
batch_size = 10 

def getTrainBatchImages(cat_image_names, dog_image_names):
    images = []
    gt_1_hot_vec = []
    for i in xrange(batch_size):
        images.append(utils.load_image(str(train_dir) + 'cat/' + str(cat_image_names[i])))
        gt_1_hot_vec.append([1,0])
        images.append(utils.load_image(str(train_dir) + 'dog/' + str(dog_image_names[i])))
        gt_1_hot_vec.append([0,1])

    return images, gt_1_hot_vec

def getTestBatchImages(cat_image_names, dog_image_names):
    images = []
    gt_1_hot_vec = []
    for i in xrange(batch_size):
        images.append(utils.load_image(str(test_dir) + 'cat/' + str(cat_image_names[i])))
        gt_1_hot_vec.append([1,0])
        images.append(utils.load_image(str(test_dir) + 'dog/' + str(dog_image_names[i])))
        gt_1_hot_vec.append([0,1])

    return images, gt_1_hot_vec    

with tf.device('/cpu:0'):
    with tf.Session() as sess:        
        images = tf.placeholder("float", [20, 224, 224, 3])
        gt = tf.placeholder(dtype=tf.float32, shape=[batch_size*2,2])

        vgg = vgg16_cat_and_dog.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images, gt)
        sess.run(tf.global_variables_initializer())

        prob = vgg.prob

        train_step = vgg.train(prob, gt)

        test_step = vgg.test(prob, gt)

        cat_image_names_train = os.listdir(str(train_dir) + 'cat/')
        dog_image_names_train = os.listdir(str(train_dir) + 'dog/')

        for epoch in xrange(max_epochs):
            random.shuffle(cat_image_names_train)
            random.shuffle(dog_image_names_train)

            # for batch in xrange(len(dog_image_names_train)/batch_size):
            for batch in xrange(100):
                batch_images, gt_1_hot_vec = \
                    getTrainBatchImages(cat_image_names_train[batch * batch_size : (batch * batch_size) + batch_size], \
                                dog_image_names_train[batch * batch_size : (batch * batch_size) + batch_size])

                opt_var_list = sess.run(train_step, feed_dict={images: batch_images, gt: gt_1_hot_vec})
                out_f = open('out_batch_file.txt', 'a+')
                out_f.write(str(epoch) + ' ' + str(batch) + '\n')
                out_f.close()                                    
                # print('epoch-batch: ', epoch, batch)
                # print('var list len: ', len(opt_var_list))
                # print('var_list: ', opt_var_list[0].shape, opt_var_list[1].shape, \
                #                 opt_var_list[2].shape, opt_var_list[3].shape)

            print('testing....')
            cat_image_names_test = os.listdir(str(test_dir) + 'cat/')
            dog_image_names_test = os.listdir(str(test_dir) + 'dog/')
            random.shuffle(cat_image_names_test)
            random.shuffle(dog_image_names_test)            

            tot_accuracy = 0.0
            # for batch in xrange(len(dog_image_names_test)/batch_size):
            for batch in xrange(20):
                batch_images_test, gt_1_hot_vec_test = \
                    getTestBatchImages(cat_image_names_test[batch * batch_size : (batch * batch_size) + batch_size], \
                            dog_image_names_test[batch * batch_size : (batch * batch_size) + batch_size])        

            # batch_images_test, gt_1_hot_vec_test = \
            #         getTestBatchImages(cat_image_names_test[0:10], dog_image_names_test[0:10])
                accuracy = sess.run(test_step, feed_dict={images: batch_images_test, gt: gt_1_hot_vec_test})
                tot_accuracy = tot_accuracy + accuracy
                # print('accuracy: ', accuracy)

            out_f = open('out_test_file.txt', 'a+')
            out_f.write(str(epoch) + ' ' + str(float(tot_accuracy/20)) + '\n')
            out_f.close()                    

            print('avg accuracy: ', tot_accuracy/20)
