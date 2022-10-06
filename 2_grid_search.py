from tensorflow.config.experimental import set_visible_devices, set_memory_growth, list_physical_devices, list_logical_devices
gpus, gpu_no = list_physical_devices('GPU'), 0
if gpus:
    try:
        set_visible_devices(gpus[gpu_no], 'GPU')
        set_memory_growth(gpus[gpu_no], enable=True)
        
        logical_gpus = list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

import os
data_path = 'data'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')
eval_path = os.path.join(data_path, 'eval')

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def define_model():
    
    input_ = Input(shape=(96, 96, 3))
    conv_base = input_

    for n_filters in [64, 64, 128, 128]:
        conv_base = Conv2D(n_filters, 3, activation="relu", padding="same")(conv_base)
        conv_base = MaxPooling2D(2)(conv_base)

    x = Flatten()(conv_base)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu")(x)    
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_, outputs=output)
    
    return model

def train_eval_model(model, train_path, eval_path):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    b_size = 32
      
    datagen_train = ImageDataGenerator(validation_split=0.2, rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen_val = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    it_train = datagen_train.flow_from_directory(train_path, target_size=(96, 96), class_mode='sparse', subset='training', batch_size=b_size, shuffle=True)
    it_val = datagen_val.flow_from_directory(train_path, target_size=(96, 96), class_mode='sparse', subset='validation', batch_size=b_size, shuffle=False)

    model.fit(it_train, steps_per_epoch=len(it_train), epochs=15, validation_data=it_val, validation_steps=len(it_val))

    datagen_eval = ImageDataGenerator(rescale=1./255)
        
    it_eval = datagen_eval.flow_from_directory(eval_path, target_size=(96, 96), class_mode='sparse', batch_size=b_size, shuffle=False)
    _, accuracy = model.evaluate(it_eval, steps=len(it_eval))
    print('Accuracy: %.2f' % (accuracy*100))
    
model = define_model()
model.summary()
train_eval_model(model, train_path, eval_path)