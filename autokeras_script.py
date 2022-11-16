import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow_addons as tfa
import tensorflow as tf
import autokeras as ak
import argparse
import warnings
import keras_tuner
warnings.filterwarnings('ignore')

spectrograms = ['spectrogram', 'mel', 'mfcc', 'mtf', 'mtm']

def normImages(X):
    for i, image in enumerate(X):
        X_temp = X[i].reshape(X[i].shape[0:2])
        scaler = MinMaxScaler(feature_range=(0.0,1.0))
        X[i] = scaler.fit_transform(X_temp).reshape(X_temp.shape+(1,))
    return X

def train_val_test_split(balanced, normalized, spectrogram, size):
    # spectrogram = ['spectrogram']
    
    X_train = []
    X_test = []
    X_val = []

    y_flag = 0 # y's has not been obtained yet 
    root_dir = '/home/prime/polivares/powerlinefaults/MTF_signals/'
    
    for sp in spectrogram:
        X_full = np.load(f"{root_dir}dataset_{sp}_{size}/full/images_full.npy")
        
        if not y_flag:
            y_flag = 1
            y_full = np.load(f"{root_dir}dataset_{sp}_{size}/full/labels_full.npy").reshape(-1)
            if balanced: # getting balanced data from index
                # Index 1, partial discharge
                index_1 = np.where(y_full==1)[0]
                len_index_1 = len(index_1)
#                 index_train_1, index_val_1, index_test_1 = index_1[:len_index_1//3], index_1[len_index_1//3:2*len_index_1//3], index_1[2*len_index_1//3:4*len_index_1//3]
                index_train_1, index_val_1, index_test_1 = index_1[:len_index_1//3], index_1[len_index_1//3:2*len_index_1//3], index_1[2*len_index_1//3:4*len_index_1//3]
    
                # Index 0, non partial discharge
                index_0 = np.where(y_full==0)[0]
#                 index_train_0, index_val_0, index_test_0 = index_0[:len_index_1//3], index_0[len_index_1//3:2*len_index_1//3], index_0[2*len_index_1//3:4*len_index_1//3]
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
                len_data = int(len(y_full)*0.70)
                len_train = int(0.8*len_data)
                len_val = len_data-len_train
                len_test = len(y_full)-len_data
                                
                # Obtaining index
                index_train, index_val, index_test = index_full[:len_train], index_full[len_train:len_data], index_full[len_data:]
            
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

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', type=str, required=True)
    parser.add_argument('-sz', '--size', type=str, required=True)
    parser.add_argument('-mt', '--max_trials', type=int, default=10, required=False)
    parser.add_argument('-s', '--spectrograms', nargs='+', default=['spectrogram'])
    parser.add_argument('-b', '--balanced',
                    action='store_true', required=False)
    parser.add_argument('-n', '--normalized',
                    action='store_true', required=False)

    args = parser.parse_args()
    assert set(args.spectrograms).issubset(set(spectrograms))
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(balanced=args.balanced,
                                                                            normalized=args.normalized,
                                                                            spectrogram=args.spectrograms,
                                                                            size=args.size)

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

    project_name = args.project_name

    clf = ak.ImageClassifier(overwrite=False,  
                        metrics=METRICS,
                        project_name=project_name,
                        seed=42,
                        max_trials=args.max_trials, 
                        objective=keras_tuner.Objective('val_recall', direction='max')
                       )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{project_name}')
    clf.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            batch_size=20,
            callbacks=[tensorboard_callback]
        )
    # Export best model
    model = clf.export_model()

    try:
        model.save(f"{project_name}/model_autokeras", save_format="tf")
    except Exception:
        model.save(f"{project_name}/model_autokeras.h5")
        
if __name__ == '__main__':
    main()
