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
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import ast

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from timeit import default_timer as timer
from contextlib import redirect_stdout

IMG_WIDTH_HEIGHT = 96

def exec_time(start, end):
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

def print_log(text, file):
    print(f'{text}\n')
    with open(f'{file}', 'a') as f:
        print(f'{text}\n', file=f)
        
def save_model_info(model, cfg, models_path):
    with open(f"{models_path}/model_infos/{cfg['name']}_summary.txt", "w") as f:
        with redirect_stdout(f):
            model.summary()
        plot_model(model, to_file=f"{models_path}/model_infos/{cfg['name']}.png", show_shapes=True, dpi=80)

def learning_curves(histories, i, accuracy, model_accs, pdf):
    fig = plt.figure(figsize=(10, 4))
    
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    for hist in histories:
        hist_dict = hist.history
        epochs = [x+1 for x in hist.epoch]
        
        ax1.plot(epochs, hist_dict['loss'], color="blue", linewidth="0.8")
        ax1.plot(epochs, hist_dict['val_loss'], color="red", linewidth="0.8")
    
        ax2.plot(epochs, hist_dict['accuracy'], color="blue", linewidth="0.8")
        ax2.plot(epochs, hist_dict['val_accuracy'], color="red", linewidth="0.8")
    
    ax1.set_title('Opetus- ja validointivirhe')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Opetusvirhe', 'Validointivirhe'], loc='upper right')
    
    ax2.set_title('Opetus- ja validointitarkkuus')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Opetustarkkuus', 'Validointitarkkuus'], loc='upper right')
    
    plt.suptitle(f"cfg nro. {i}: >>> {accuracy*100:.5f} {str(model_accs)}")
    plt.tight_layout()
        
    pdf.savefig(fig)
    plt.close(fig)

def build_model(input_shape, cfg):   
    input_ = Input(shape=(input_shape))
    conv_base = input_

    #Piilokerrosten rakennus
    for n_filters in cfg['n_filters']:
        conv_base = Conv2D(n_filters, 3, activation=cfg['act'], padding="same")(conv_base)
        conv_base = MaxPooling2D(2)(conv_base)

    x = Flatten()(conv_base)
    x = Dropout(cfg['drop_out'])(x)
    x = Dense(512, activation="relu")(x)    
    output = Dense(cfg['num_classes'], activation="sigmoid")(x)
    model = Model(inputs=input_, outputs=output)
    
    return model

def train_model(it_train, it_val, cfg, models_path):
    model = build_model((IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, 3), cfg)
    
    if cfg['optimizer'] == 'Adam':
        sgd = Adam(amsgrad=True, learning_rate=cfg['lr'])
    elif cfg['optimizer'] == 'SGD':
        sgd = SGD(learning_rate=cfg['lr'])
    elif cfg['optimizer'] == 'Adagrad':
        sgd = Adagrad(learning_rate=cfg['lr'])        
     
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    checkpoint_dir = os.path.sep.join([models_path, 'tmp'])
    checkpoint_filepath = os.path.sep.join([checkpoint_dir, 'checkpoint'])
    
    cbs_list = [
        ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', mode='max', save_weights_only=True, verbose=0),
        CSVLogger(os.path.sep.join([models_path, 'training.log']), append=True)
    ]
    
    history = model.fit(it_train, epochs=cfg['epochs'], validation_data=it_val, callbacks=cbs_list)
    
    open(os.path.sep.join([models_path, 'training.log']), 'a').write(f"\n")
    
    model.load_weights(checkpoint_filepath)
    shutil.rmtree(checkpoint_dir)
    
    return model, history

def evaluate_models(models, it_eval, cfg):
    #OOM esto, kun käytetään CPU:ta GPU:n sijaan
    with tf.device('/cpu:0'):    
        X_test_eval = np.concatenate([it_eval .next()[0] for i in range(it_eval .__len__())])
        y_test_eval = np.concatenate([it_eval .next()[1] for i in range(it_eval .__len__())])

        all_preds = [model.predict(X_test_eval, batch_size=cfg['b_size'])[:,0] for model in models]

        model_accs = [accuracy_score(y_test_eval, (preds > 0.5).astype(int)) for preds in all_preds]
        model_accs = np.array(model_accs).round(6)

        weighted_preds = (np.average(all_preds, axis=0, weights=model_accs) > 0.5).astype(int)
        weighted_acc = accuracy_score(y_test_eval, weighted_preds)

        print(f"Evaluation accuracy (model accuracies): {str(model_accs)}\n")
        print(f"Evaluation accuracy (weighted accuracy): {weighted_acc*100:.3f}\n")
        return model_accs, weighted_acc

def grid_search(cfgs, models_path, pdf, n_repeats):    
    avg_scores = []
    best_scores = []
    
    print(f"Yhteensä {len(cfgs)} konfiguraatiota, {n_repeats} toistolla, aloitetaan...")
    for i, cfg in cfgs.items():
        cfg['name'] = f"2D-CNN-{i}"
        print_log(f"Konfiguraatio {i}: {cfg}", f"{models_path}/cfgs.txt")
        open(os.path.sep.join([models_path, 'training.log']), 'a').write(f"cfg: {i}\n")
        
        it_train, it_val, it_eval = load_data(cfg)
        
        models_histories = [train_model(it_train, it_val, cfg, models_path) for _ in range(n_repeats)]
        models = [model for model, history in models_histories]
        histories = [history for model, history in models_histories]
        save_model_info(models[0], cfg, models_path)
        
        model_accs, weighted_acc = evaluate_models(models, it_eval, cfg)
        learning_curves(histories, i, weighted_acc, model_accs, pdf)
        
        avg_scores.append((i, weighted_acc, model_accs, models))
        
        for j, (acc, model) in enumerate(zip(model_accs, models), 1):
            best_scores.append((f"{i}_v{j}", acc, model)) 
        
        if len(avg_scores) > 10:
            avg_scores.sort(key=lambda tup: tup[1], reverse=True)
            del avg_scores[10:]
        
        if len(best_scores) > 10:
            best_scores.sort(key=lambda tup: tup[1], reverse=True)
            del best_scores[10:]                        
    
    avg_scores.sort(key=lambda tup: tup[1], reverse=True)
    best_scores.sort(key=lambda tup: tup[1], reverse=True)
    
    pdf.close()
    
    return avg_scores, best_scores

def load_data(cfg):
    data_path = cfg['data_path'] #Käytetäänkö täyttä vai rajattua datasettiä
    train_path = os.path.join(data_path, 'train')
    test_eval_path = os.path.join(data_path, 'test_eval')
    
    b_size = cfg['b_size']
    
    #Datan augmentointi
    datagen_train = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.15)
    datagen_val = ImageDataGenerator(validation_split=0.15, rescale=1./255)
    datagen_test_eval = ImageDataGenerator(rescale=1./255)

    it_train = datagen_train.flow_from_directory(train_path, target_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT), class_mode='binary', batch_size=b_size, shuffle=True, subset="training")
    it_val = datagen_val.flow_from_directory(train_path, target_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT), class_mode='binary', batch_size=b_size, shuffle=True, subset="validation")
    it_test_eval = datagen_test_eval.flow_from_directory(test_eval_path, target_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT), class_mode='binary', batch_size=b_size, shuffle=False)
    
    return it_train, it_val, it_test_eval
    
time = datetime.now().strftime('%Y%m%dT%H%M')
models_path = f"models_{time}"

if not os.path.exists(models_path):
    os.mkdir(models_path)

os.mkdir(os.path.sep.join([models_path, "model_infos"]))
os.mkdir(os.path.sep.join([models_path, "models"]))

pdf = PdfPages(os.path.join(models_path, "learning_curves.pdf"))

param_grid = ast.literal_eval(open("param_grid.txt", "r").read())
cfgs = {i: cfg for i, cfg in enumerate(list(ParameterGrid(param_grid)), 1)}
                          
n_repeats = 2

start = timer()
avg_scores, best_scores = grid_search(cfgs, models_path, pdf, n_repeats)
end = timer()

write_to_file = ""
for i, avg_acc, model_accs, models in avg_scores:
    print(i, avg_acc, model_accs, models)
    write_to_file = write_to_file + f"{i}: {avg_acc*100:.3f} {model_accs}\n"
    
    for j, model in enumerate(models, 1):
        filepath = os.path.sep.join([models_path, 'models', f'{i}_v{j}.hdf5'])
        print(filepath)
        model.save(filepath)
    print()

print_log(write_to_file, f"{models_path}/avg_results.txt")

write_to_file = ""
for name, acc, model in best_scores:
    print(name, acc)
    write_to_file = write_to_file + f"{name}: {acc*100:.3f}\n"
    
    if f'{name}.hdf5' not in os.listdir(models_path):
        filepath = os.path.sep.join([models_path, 'models', f'{name}.hdf5'])
        print(filepath)
        model.save(filepath)
    print()

print_log(write_to_file, f"{models_path}/best_results.txt")

print("Time elapsed (hh:mm:ss):", exec_time(start, end))