import librosa
import torchaudio
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd

# Reusable code
def get_label(file_name):
  '''
  Function to retrieve output labels from filenames
  '''
  if 'ROC' in file_name:
    label=0
  elif 'LES' in file_name:
    label=1
  elif 'DCB' in file_name:
    label=2
  elif 'PRV' in file_name:
    label=3
  elif 'VLD' in file_name:
    label=4
  elif 'DTA' in file_name:
    label=5
  else:
    raise ValueError('invalid file name')
  return label

def train_test_preprocess(func, feat_name: str, args: dict):
    train_files = glob('./project_data/train/*.wav')
    train_files.sort()
    train_feat=[]
    train_label=[]
    for wav in tqdm(train_files):
        train_feat.append(func(wav, **args))
        train_label.append(get_label(wav))

    test_clean_files = glob('./project_data/test_clean/*.wav')
    test_clean_files.sort()
    test_clean_feat=[]
    test_clean_label=[]
    for wav in tqdm(test_clean_files):
        test_clean_feat.append(func(wav, **args))
        test_clean_label.append(get_label(wav))

    test_noisy_files = glob('./project_data/test_noisy/*.wav')
    test_noisy_files.sort()
    test_noisy_feat=[]
    test_noisy_label=[]
    for wav in tqdm(test_noisy_files):
        test_noisy_feat.append(func(wav, **args))
        test_noisy_label.append(get_label(wav))

    feat_names=[f'{feat_name}_' +str(n) for n in range(len(train_feat[0]))]
    train_feat_df = pd.DataFrame(data=np.stack(train_feat), columns=feat_names)
    y_train=np.stack(train_label)
    test_clean_feat_df = pd.DataFrame(data=np.stack(test_clean_feat), columns=feat_names)
    y_test_clean=np.stack(test_clean_label)
    test_noisy_feat_df = pd.DataFrame(data=np.stack(test_noisy_feat), columns=feat_names)
    y_test_noisy=np.stack(test_noisy_label)
    return train_feat_df, y_train, test_clean_feat_df, y_test_clean, test_noisy_feat_df, y_test_noisy

def save_features(data, name: str):
    train_feat_df, y_train, test_clean_feat_df, y_test_clean, test_noisy_feat_df, y_test_noisy = data
    directory = 'saved_features'
    train_feat_df.to_csv(f'{directory}/{name}_train.csv')
    test_clean_feat_df.to_csv(f'{directory}/{name}_test_clean.csv')
    test_noisy_feat_df.to_csv(f'{directory}/{name}_test_noisy.csv')
    return

# Feature Extraction
def first_deriv(audio_file, downsample):
    y, sr = librosa.load(audio_file, sr=None)
    new_sr = sr//downsample
    audio = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
    feat_out = librosa.feature.delta(audio,order=1)
    return feat_out

def second_deriv(audio_file, downsample):
    y, sr = librosa.load(audio_file, sr=None)
    new_sr = sr//downsample
    audio = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
    feat_out = librosa.feature.delta(audio,order=2)
    return feat_out


def mfcc(audio_file, n_mfcc=13, fderiv=False, sderiv=False):
    audio,fs = torchaudio.load(audio_file)
    audio = audio.numpy().reshape(-1)

    # replace the following features with your own
    mfccs = librosa.feature.mfcc(y=audio,sr=fs,n_mfcc=n_mfcc)
    feat_out = np.mean(mfccs,axis=1)
    if fderiv:
        out = librosa.feature.delta(mfccs,order=1)
        out = np.mean(out,axis=1)
        feat_out = np.concatenate((feat_out,out),axis=None)
    if sderiv:
        out = librosa.feature.delta(mfccs,order=2)
        out = np.mean(out,axis=1)
        feat_out = np.concatenate((feat_out,out),axis=None)
    return feat_out