import scipy as sp
from librosa.feature import melspectrogram, mfcc
from sklearn.feature_extraction import img_to_graph
from sklearn.preprocessing import MinMaxScaler
from pyts.image import MarkovTransitionField
import tsia.markov
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2

spectrograms = ['spectrogram', 'mel', 'mfcc', 'mtf', 'mtm']
image_types = ['full', 'train', 'test', 'val']

def signal2spectrogram(signal, fs, spectrogram = 'spectrogram', img_size=(256, 256), window=('tukey', 0.25)):
    assert spectrogram in spectrograms
    # Get spectrogram from signal
    if spectrogram == 'spectrogram':
        _, _, Sxx = sp.signal.spectrogram(signal, fs, window=window)
    elif spectrogram == 'mel':
        Sxx = melspectrogram(signal.to_numpy().astype(float), fs)
    elif spectrogram == 'mfcc':
        Sxx = mfcc(signal.to_numpy().astype(float), fs)
    elif spectrogram == 'mtf':
        transformer = MarkovTransitionField(30)
        Sxx = transformer.fit_transform(signal.to_numpy().astype(float).reshape(1,-1))[0]
    elif spectrogram == 'mtm':
        X_binned, bin_edges = tsia.markov.discretize(signal, 30, strategy='uniform')
        X_mtm = tsia.markov.markov_transition_matrix(X_binned)
        Sxx = tsia.markov.markov_transition_probabilities(X_mtm)

    # Transform spectrogram to image range (0, 255)
    scaler = MinMaxScaler(feature_range=(0,255))
    Sxx = scaler.fit_transform(Sxx)
    # Resize image
    img = np.array(Image.fromarray(Sxx).resize(img_size))

    return img

def signals2images(signals, fs, spectrogram = 'spectrogram', img_size = (256, 256), window=('tukey', 0.25)):
    assert spectrogram in spectrograms
    images = []
    for col in tqdm(signals.columns):
        # Dask dataframe
        signal = signals[str(col)].compute()
        images.append(signal2spectrogram(signal, fs, 
                                         spectrogram=spectrogram, 
                                         img_size=img_size,
                                         window=window))
    img_reshape = (len(signals.columns),) + img_size + (1,)
    images = np.array(images).reshape(img_reshape)
    return images

def saveImages(images, labels, root_dir = '', spectrogram = 'spectrogram', image_type='full', saveViz=False):
    assert spectrogram in spectrograms
    assert image_type in image_types
    image_path = os.path.join(root_dir, 'dataset_'+spectrogram, image_type)
    image_file = 'images_'+image_type+'.npy'
    label_file = 'labels_'+image_type+'.npy'

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    np.save(os.path.join(image_path,image_file), images)
    np.save(os.path.join(image_path,label_file), labels)

    if saveViz:
        for image, i in zip(images, range(labels.size)):
            image_file = f'image_{image_type}_{i}.jpg'
            cv2.imwrite(os.path.join(image_path,image_file), image)


    





