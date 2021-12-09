import torch
import soundfile as sf
from sklearn import svm
import numpy as np
import pandas as pd
import utils
import glob

def svm_regression(embeddings_train, labels_train, embeddings_test, labels_test):
    clf = svm.SVR(kernel='linear')

    clf.fit(embeddings_train, labels_train)

    pred = clf.predict(embeddings_test)
    
    r2_score = clf.score(embeddings_test, labels_test)
    
    return pred, r2_score


def extract_features(model, wav_paths):
    X, y = [], []

    device = next(model.parameters()).device
    
    for wav_path in wav_paths:
        audio, sr = sf.read(wav_path, dtype='float32')
        audio_len = len(audio) / sr
        y.append(audio_len)
        
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        output = model(audio)
        feat = output.detach().cpu().squeeze().numpy()
        X.append(feat)
        
    return X, y
            

def probe_audio_duration(model, wav_paths_train, wav_paths_test):
    X_train, y_train = extract_features(model, wav_paths_train)
    X_test, y_test = extract_features(model, wav_paths_test)

    pred, r2_score = svm_regression(X_train, y_train, X_test, y_test)
    
    df = pd.DataFrame(list(zip(y_test, pred)), columns=['duration', 'pred_duration'])
    
    return df, r2_score


if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    model = utils.load_wav2vec2(mode='base', device=device)
    wav_paths_train = glob.glob("../SpeakerRecognition/data/data/TRAIN/*/*/*.wav")
    wav_paths_test = glob.glob("../SpeakerRecognition/data/data/TEST/*/*/*.wav")
    
    pred, score = probe_audio_duration(model, wav_paths_train, wav_paths_test)