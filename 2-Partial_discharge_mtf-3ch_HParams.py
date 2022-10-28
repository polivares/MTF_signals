#!/usr/bin/env python
# coding: utf-8

# # Loading data

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from tensorboard.plugins.hparams import api as hp

# Dataset hyper parameters
HP_BALANCED = hp.HParam('balanced', hp.Discrete([0, 1])) # 0 unbalanced, 1 balanced
HP_NORM_IMAGES = hp.HParam('norm_images', hp.Discrete([0, 1]))

# Network hyper parameters
HP_NETWORKS = hp.HParam('network', hp.Discrete(['Xception',
                                            'VGG16',
                                            'VGG19',
                                            'ResNet50',
                                            'ResNet50V2',
                                            'ResNet101',
                                            'ResNet101V2',
                                            'ResNet152',
                                            'ResNet152V2',
                                            'InceptionV3',
                                            'InceptionResNetV2',
                                            'MobileNet',
                                            'MobileNetV2',
                                            'DenseNet121',
                                            'DenseNet169',
                                            'DenseNet201',
                                            'NASNetMobile',
                                            'NASNetLarge',
                                            'EfficientNetB0',
                                            'EfficientNetB1',
                                            'EfficientNetB2',
                                            'EfficientNetB3',
                                            'EfficientNetB4',
                                            'EfficientNetB5',
                                            'EfficientNetB6',
                                            'EfficientNetB7',
                                            'EfficientNetV2B0',
                                            'EfficientNetV2B1',
                                            'EfficientNetV2B2',
                                            'EfficientNetV2B3',
                                            'EfficientNetV2S',
                                            'EfficientNetV2M',
                                            'EfficientNetV2L',
                                            'ConvNeXtTiny',
                                            'ConvNeXtSmall',
                                            'ConvNeXtBase',
                                            'ConvNeXtLarge',
                                            'ConvNeXtXLarge' ])) 


HP_ACTIVATION_FUNCTIONS = hp.HParam('act_functions', hp.Discrete(['relu', 'selu', 'tanh']))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64, 128]))
HP_EARLY_STOP = hp.HParam('early_stop', hp.Discrete([3, 5, 7, 10, 15]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))


# In[ ]:


import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        result_metrics = train_test_model(hparams)
        tf.summary.scalar(METRIC_NAMES, result_metrics, step=1)



from tensorflow.keras import datasets, layers, models

# Function for normalizing images
def normImages(X):
    for i, image in enumerate(X):
        max_n = np.max(image)
        image /= max_n
        X[i] = np.abs(image)
    return X

# Obtaining training, validation and test data
def train_val_test_split(balanced, normalized):
    spectrogram = ['spectrogram', 'mel', 'mtf']
    
    X_train = []
    X_test = []
    X_val = []

    y_flag = 0 # y's has not been obtained yet 
    root_dir = "/home/polivares/scratch/Datasets/PowerLineFaults/"
    # root_dir = '/home/polivares/Dropbox/Work/PostDoc/PowerLineFaults/'
    
    for sp in spectrogram:
        X_full = np.load(f"{root_dir}dataset_{sp}/full/images_full.npy")
        
        
        if not y_flag:
            y_flag = 1
            
            y_full = np.load(f"{root_dir}dataset_{sp}/full/labels_full.npy").reshape(-1)
            if balanced: # getting balanced data from index
                # Index 1, partial discharge
                index_1 = np.where(y_full==1)[0]
                len_index_1 = len(index_1)
                index_train_1, index_val_1, index_test_1 = index_1[:len_index_1//3], index_1[len_index_1//3:2*len_index_1//3], index_1[2*len_index_1//3:]

                # Index 0, non partial discharge
                index_0 = np.where(y_full==0)[0]
                index_train_0, index_val_0, index_test_0 = index_0[:len_index_1//3], index_0[len_index_1//3:2*len_index_1//3], index_0[2*len_index_1//3:]

                # Obtaining index
                index_train = np.concatenate([index_train_0, index_train_1])
                np.random.shuffle(index_train)
                index_val = np.concatenate([index_val_0, index_val_1])
                np.random.shuffle(index_val)
                index_test = np.concatenate([index_test_0, index_test_1])
                np.random.shuffle(index_test)

            else: # Unbalanced data, similar to the original from index
                index_full = np.arange(len(y_full))
                np.random.shuffle(index_full)
                len_index = 3000
                
                # Obtaining index
                index_train, index_val, index_test = index_full[:len_index], index_full[len_index:2*len_index], index_full[2*len_index:]
            
            y_train = y_full[index_train]
            y_val = y_full[index_val]
            y_test = y_full[index_test]
            
            del y_full
        
        if normalized:
            X_full = normImages(X_full)
        
        X_train.append(X_full[index_train])
        X_val.append(X_full[index_val])
        X_test.append(X_full[index_test])
        
        del X_full
        
    X_train_c = np.concatenate(X_train, axis=3) 
    X_val_c = np.concatenate(X_val, axis=3)
    X_test_c = np.concatenate(X_test, axis=3)
    
    return X_train_c, y_train, X_val_c, y_val, X_test_c, y_test


# X_train_c, y_train, X_val_c, y_val, X_test_c, y_test = train_val_test_split(balanced=0,normalized=0)


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

def train_test_model(hparams, METRICS, data):
    # Model creation
    X_train, y_train, X_val, y_val, X_test, y_test = data
    print("Model creation")
    base_model = getattr(tf.keras.applications, hparams[HP_NETWORKS])(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(64, (3,3), activation=hparams[HP_ACTIVATION_FUNCTIONS])(base_model.output)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION_FUNCTIONS])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    # avg = tf.keras.layers.GlobalAveragePooling2D()()
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False # Esto impide que las capas se re entrenen
    
    # Compile model
    print("Model compile")
    early_stopping = EarlyStopping(patience = hparams[HP_EARLY_STOP])
    model.compile(loss = 'bce', optimizer = hparams[HP_OPTIMIZER], metrics=METRICS)
    
    
    # Fitting training
    print("Fitting training")
    history_model = model.fit(X_train, y_train, epochs=1000, 
                            validation_data=(X_val, y_val),
                            batch_size=10,
                            callbacks=[early_stopping])
    # Evaluation on test
    print("Evaluation on test")
    results = model.evaluate(X_test, y_test)
    
    # Returning metrics results
    print("Returning metrics results")
    return results


# In[ ]:


# Run evaluation with hparams
def run(run_dir, hparams, data):
    METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'), 
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    METRICS_NAMES = [
        'loss',
        'tp',
        'fp',
        'tn',
        'fn',
        'accuracy',
        'precision',
        'recall',
        'auc',
        'prc'
    ]

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        # Obtaining data 
        # print("Obtaining data")
        # data = train_val_test_split(hparams[HP_BALANCED], hparams[HP_NORM_IMAGES])
        results = train_test_model(hparams, METRICS, data)
    
        for name, metric in zip(METRICS_NAMES, results):
            print(f"Summary: metric {name} value {metric}")
            tf.summary.scalar(name, metric, step=1)


# In[ ]:


if __name__=='__main__':

    session_num = 0

    # Obtaining data 
    print("Obtaining data")
    # data = train_val_test_split(hparams[HP_BALANCED], hparams[HP_NORM_IMAGES])
    data = train_val_test_split(0, 0)

    for network in HP_NETWORKS.domain.values:
        for act_func in HP_ACTIVATION_FUNCTIONS.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for dropout in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    for early_stop in HP_EARLY_STOP.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            hparams = {
                                HP_BALANCED : 0,
                                HP_NORM_IMAGES: 0,
                                HP_NETWORKS: network,
                                HP_ACTIVATION_FUNCTIONS: act_func,
                                HP_NUM_UNITS: num_units,
                                HP_DROPOUT: dropout,
                                HP_EARLY_STOP: early_stop,
                                HP_OPTIMIZER: optimizer
                            }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run('logs/hparam_tuning/' + run_name, hparams, data)
                            session_num += 1

