import numpy as np
import tensorflow as tf
import random
import vgg16_cat_and_dog_svm
import utils_svm
import os
import time
from sklearn import svm

train_dir = "/g/kumarsha/project/data/training/"
test_dir = "/g/kumarsha/project/data/testing/"
batch_size = 50 

def getTrainBatchImages(cat_image_names, dog_image_names):
    images = []
    y = []
    # for i in xrange(batch_size):
    for i in xrange(len(cat_image_names)):
        images.append(utils_svm.load_image(str(train_dir) + 'cat/' + str(cat_image_names[i])))
        y.append(1)
        images.append(utils_svm.load_image(str(train_dir) + 'dog/' + str(dog_image_names[i])))
        y.append(0)

    return images, y

def getTestBatchImages(cat_image_names, dog_image_names):
    images = []
    y = []
    # for i in xrange(batch_size):
    for i in xrange(len(cat_image_names)):
        images.append(utils_svm.load_image(str(test_dir) + 'cat/' + str(cat_image_names[i])))
        y.append(1)
        images.append(utils_svm.load_image(str(test_dir) + 'dog/' + str(dog_image_names[i])))
        y.append(0)

    return images, y    

with tf.device('/cpu:0'):
    with tf.Session() as sess:     
        start_time = time.time()

        cat_image_names_train = os.listdir(str(train_dir) + 'cat/')
        dog_image_names_train = os.listdir(str(train_dir) + 'dog/')
        random.shuffle(cat_image_names_train)
        random.shuffle(dog_image_names_train)                

        cat_image_names_test = os.listdir(str(test_dir) + 'cat/')
        dog_image_names_test = os.listdir(str(test_dir) + 'dog/')
        random.shuffle(cat_image_names_test)
        random.shuffle(dog_image_names_test)                    

        images = tf.placeholder("float", [100, 224, 224, 3])

        vgg = vgg16_cat_and_dog_svm.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        sess.run(tf.global_variables_initializer())

        relu7 = vgg.relu7

        train_x = []
        train_y = []

        start_time = time.time()

        # for batch in xrange(150):
        for batch in xrange(1):
            batch_images, y = \
                getTrainBatchImages(cat_image_names_train[batch * batch_size : (batch * batch_size) + batch_size], \
                            dog_image_names_train[batch * batch_size : (batch * batch_size) + batch_size])        

            train_activations = sess.run(relu7, feed_dict={images: batch_images})
            # print('train_activations: ', train_activations.shape)   

            for x in xrange(train_activations.shape[0]):
                train_x.append(train_activations[x])
                train_y.append(y[x])

            duration = time.time() - start_time
            # print('time taken for batch ' + str(batch) + ': ' + str(round(duration/float(60),2))) 
            out_f = open('out_train_file.txt', 'a+')
            out_f.write('time taken for batch ' + str(batch) + ': ' + str(round(duration/float(60),2)) + '\n')
            out_f.close()                                 

        out_f = open('out_train_file.txt', 'a+')
        out_f.write('train y: ' + str(train_y))
        out_f.close()        

        fit_start_time = time.time()
        print('training start time: ', fit_start_time)
        clf = svm.SVC(C=0.5, kernel='linear')
        training_output = clf.fit(train_x, train_y)  
        print('training complete in: ', str(time.time() - fit_start_time) + '\n')  

        test_x = []
        test_y = []        
        # for batch in xrange(50):
        for batch in xrange(1):
            batch_images_test, y = \
                getTestBatchImages(cat_image_names_test[batch * batch_size : (batch * batch_size) + batch_size], \
                        dog_image_names_test[batch * batch_size : (batch * batch_size) + batch_size])                                            

            test_activations = sess.run(relu7, feed_dict={images: batch_images_test})  

            for x in xrange(test_activations.shape[0]):
                test_x.append(test_activations[x])
                test_y.append(y[x])

            duration = time.time() - start_time
            out_f = open('out_test_file.txt', 'a+')
            out_f.write('time taken for batch ' + str(batch) + ': ' + str(round(duration/float(60),2)) + '\n')
            out_f.close()

        print('len test x: ', len(test_x))
        print('len test y: ', len(test_y))
        test_start_time = time.time()
        print('testing start time: ', test_start_time)
        testing_output = clf.predict(test_x)
        print('testing end time: ', time.time())

        # print('testing complete in: ', str(round((time.time() - test_start_time)/float(60),2)))  
        print('testing complete in: ', str(time.time() - test_start_time))

        hit = 0.0
        for real, predicted in zip(test_y, testing_output):
            if(real == predicted):
                hit = hit + 1.0

        out_f = open('out_test_file.txt', 'a+')
        out_f.write('test y: ' + str(test_y) + '\n')
        out_f.write('testing_output: ' + str(testing_output) + '\n')
        out_f.write('zip: ' + str(zip(test_y, testing_output)) + '\n')
        out_f.close()                        

        print('hit: ', hit)
        print('len testing_output: ', len(testing_output))
        print('Accuracy: ', hit/len(testing_output))                              


