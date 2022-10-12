import scipy as sp
from librosa.feature import melspectrogram, mfcc
from sklearn.feature_extraction import img_to_graph
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

spectrograms = ['spectrogram', 'mel', 'mfcc']
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
                                         img_size=img_size),
                                         window=window)
    img_reshape = (len(signals.columns),) + img_size + (1,)
    images = np.array(images).reshape(img_reshape)
    return images

def saveImages(images, labels, root_dir = '', spectrogram = 'spectrogram', image_type='full'):
    assert spectrogram in spectrograms
    assert image_type in image_types
    image_path = os.path.join(root_dir, 'dataset_'+spectrogram, image_type)
    image_file = 'images_'+image_type+'.npy'
    label_file = 'labels_'+image_type+'.npy'

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    np.save(os.path.join(image_path,image_file), images)
    np.save(os.path.join(image_path,label_file), labels)
    





