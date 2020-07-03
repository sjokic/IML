# Based on the paper "Learning Fine-grained Image Similarity with Deep Ranking":
# https://arxiv.org/abs/1404.4661
# and the following implementation:
# https://github.com/akarshzingade/image-similarity-deep-ranking/
# which has been adjusted appropriately for this task and according to the above mentioned paper.

from ImageDataGeneratorCustom import ImageDataGeneratorCustom
import numpy as np
from skimage import transform
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time

#The number of training samples has to be a multiple of (batch size)*3 for this implementation to work. 
f=open("train_triplets.txt","r")
lines = sum(1 for line in f)
f.seek(0,0)
if(lines>=59509):
    print("Processing train_triplets.txt")
    d=f.read()
    f.close()
    a=d.split("\n")
    s="\n".join(a[:-(lines-59507)])
    f=open("train_triplets.txt","w+")
    for i in range(len(s)):
        f.write(s[i])
    f.write("\n")
    f.close()


class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory("./",
                                            batch_size=batch_size,
                                            target_size=self.target_size,shuffle=False,
                                            triplet_path='train_triplets.txt',
                                           )


def create_model():
    #Two small CNNs to extract low res features
    x1 = Input(shape=(224,224,3))
    x1_ = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(x1)
    x1_ = MaxPool2D(pool_size=(3,3), strides=(4,4), padding='same')(x1_)
    x1_ = Flatten()(x1_)
    x1_ = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x1_)

    x2 = Input(shape=(224,224,3))
    x2_ = Conv2D(96, kernel_size=(8,8), strides=(32,32), padding='same')(x2)
    x2_ = MaxPool2D(pool_size=(7,7), strides=(2,2), padding='same')(x2_)
    x2_ = Flatten()(x2_)
    x2_ = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x2_)
    
    #And one deep CNN to capture the main features
    x3 = Input(shape=(224,224,3))
    x3_ = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))(x3)
    x3_ = GlobalAveragePooling2D()(x3_)
    x3_ = Dense(4096, activation='relu')(x3_)
    x3_ = Dropout(0.4)(x3_)
    x3_ = Lambda(lambda  x_: K.l2_normalize(x_,axis=1))(x3_)
    
    #Combine ouputs of CNNs to generate final embedding
    x1_x2 = concatenate([x1_, x2_])

    x1_x2_x3 = concatenate([x1_x2, x3_])
    embedding = Dense(4096)(x1_x2_x3)
    norm_embedding = Lambda(lambda  x: K.l2_normalize(x,axis=1))(embedding)

    model = Model(inputs=[x1, x2, x3], outputs=norm_embedding)

    return model

#Triplet loss based on squared euclidean distance
def loss_func(y_true, y_pred):
    loss = tf.convert_to_tensor(0,dtype=tf.float32)
    total_loss = tf.convert_to_tensor(0,dtype=tf.float32)
    g = tf.constant(1.0,shape=[1],dtype=tf.float32)
    zero = tf.constant(0.0,shape=[1],dtype=tf.float32)
    for i in range(0,batch_size,3):
        try:
            A_embedding = y_pred[i]
            B_embedding = y_pred[i+1]
            C_embedding = y_pred[i+2]
            distanceAB = K.sum((A_embedding - B_embedding)**2)
            distanceAC = K.sum((A_embedding - C_embedding)**2)
            loss = tf.maximum(g + distanceAB - distanceAC, zero)
            total_loss = total_loss + loss
        except:
            continue
    total_loss = total_loss/(batch_size/3)
    return total_loss

#Data generator & data augmentation
dg = DataGenerator({
    "rescale": 1. / 255,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.2,
    "rotation_range": 30,
"fill_mode": 'nearest' 
}, target_size=(224, 224))

batch_size = 9
train_generator = dg.get_train_generator(batch_size)

deep_ranking = create_model()

#deep_ranking.load_weights('best.h5')

deep_ranking.compile(loss=loss_func, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))

train_steps_per_epoch = int((59508)/batch_size)
train_epochs = 6

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best.h5', monitor='loss', mode='min', verbose=1, save_weights_only=True,save_best_only=True)

print('Training...')
t1 = time.time()

deep_ranking.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=train_epochs,
                        callbacks=[es, mc]
                        )

print('Time taken for training:', time.time() - t1, 's')

print('Predicting...')
f = open('prediction.txt', 'w')
f_triplets = open('test_triplets.txt')
f_triplets_read = f_triplets.read()
triplets = f_triplets_read.split('\n')
f_triplets.close()

t1 = time.time()
for line in triplets:
    nums = line.split(' ')
    if len(nums)!=3:
        continue
    img1 = load_img('./food/'+nums[0]+'.jpg')
    img1 = img_to_array(img1).astype("float64")
    img1 = transform.resize(img1, (224, 224))
    img1 *= 1. / 255
    img1 = np.expand_dims(img1, axis = 0)
    embedding1 = deep_ranking.predict([img1, img1, img1])[0]
    
    img2 = load_img('./food/'+nums[1]+'.jpg')
    img2 = img_to_array(img2).astype("float64")
    img2 = transform.resize(img2, (224, 224))
    img2 *= 1. / 255
    img2 = np.expand_dims(img2, axis = 0)
    embedding2 = deep_ranking.predict([img2,img2,img2])[0]
    
    img3 = load_img('./food/'+nums[2]+'.jpg')
    img3 = img_to_array(img3).astype("float64")
    img3 = transform.resize(img3, (224, 224))
    img3 *= 1. / 255
    img3 = np.expand_dims(img3, axis = 0)
    embedding3 = deep_ranking.predict([img3,img3,img3])[0]
    
    distanceAB = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])
    distanceAC = sum([(embedding1[idx] - embedding3[idx])**2 for idx in range(len(embedding1))])
    
    if(distanceAB < distanceAC):
        f.write("1\n")
    else:
        f.write("0\n")
    
f.close()
print('Time taken for predicting:', time.time() - t1, 's')

