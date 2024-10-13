#!/usr/bin/env python3
"""
Task 0 - Transfer Knowledge
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Input
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_cifar():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                            test_size=0.15, 
                                            stratify=np.array(y_train), 
                                            random_state=42)

    print("train shape",x_train.shape, y_train.shape)
    print("test shape",x_test.shape, y_test.shape)
    print("val shape",x_val.shape, y_val.shape)

    y_train_ohe = pd.get_dummies(y_train[:,0]).to_numpy()
    y_val_ohe = pd.get_dummies(y_val[:,0]).to_numpy()
    y_test_ohe = pd.get_dummies(y_test[:,0]).to_numpy()

    print("hot one:",y_train_ohe.shape, y_test_ohe.shape)
    print("sample", y_train[0:5],"\n hot_sample", y_train_ohe[0:5])

    return (x_train, y_train_ohe, y_train), (x_test, y_test_ohe, y_test),(x_val, y_val_ohe, y_val) 



def augmentation(x_train,y_train_ohe, x_val, y_val_ohe):
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=30, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2, 
                                    horizontal_flip = 'true')
    train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, 
                                        batch_size=32, seed=1)
                                        
    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, 
                                    batch_size=32, seed=1)  

    return train_generator, val_generator

def preprocess_data(X, Y):
    x_train_scaled = x_train.astype('float32')
    x_train_scaled = x_train_scaled / 255
    x_test_scaled = x_test.astype('float32')
    x_test_scaled = x_test_scaled / 255

    return x_train_scaled, Y

(x_train, y_train_ohe, y_train), (x_test, y_test_ohe, y_test), (x_val, y_val_ohe, y_val) = prepare_cifar()
train_generator, val_generator = augmentation(x_train,y_train_ohe, x_val, y_val_ohe )

def resize_image(tensor):
    return tf.image.resize(tensor, (299, 299))

input_tensor = Input(shape=(32, 32, 3))  # Supposons que les images ont 3 canaux (par exemple, RGB)
resize_layer = Lambda(resize_image)(input_tensor)

base_inception = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=resize_layer,
    classifier_activation="softmax")

out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(2048, activation='relu')(out)
out = Dropout(rate=0.3)(out)
out = Dense(2048, activation='relu')(out)
out = Dropout(rate=0.3)(out)
predictions = Dense(10, activation='softmax')(out)

model = Model(inputs=input_tensor, outputs=predictions)  

base_inception.trainable = False

for layer in base_inception.layers[-300:]:
    layer.trainable = True

model.compile(Adam(lr=.000005), loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()

model_checkpoint_callback = ModelCheckpoint(
    filepath="./save/",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

batch_size = 512
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_test.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=80, verbose=1)

model.save('cifar10_resnnetv2_6.h5')

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
plot_history(history)

model.evaluate(x_test/255.0, y_test_ohe, batch_size=128, verbose=1)

class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicts=model.predict(x_test/255.0)
y_pred_classes = np.argmax(predicts, axis=1)
y_true = np.argmax(y_test_ohe, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 9))
c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
c.set(xticklabels=class_names, yticklabels=class_names)
plt.show()
