import os, glob, argparse
import numpy as np
import pandas as pd
import h5py, pickle
from sklearn import svm
from sklearn import metrics
import utils

def svm_classification(embeddings_train, labels_train, embeddings_test, labels_test):
#     #Create a svm Classifier
    clf = svm.SVC(kernel='linear')

    #Train the model using the training sets
    clf.fit(embeddings_train, labels_train)

    #Predict the response for test dataset
    y_pred = clf.predict(embeddings_test)

    return metrics.f1_score(labels_test, y_pred, average='macro')

#     label_set = list(set(labels_test))

#     F1s = []
#     for _ in range(100):
#         y_pred = np.random.choice(label_set, len(labels_test))
#         F1s.append(metrics.f1_score(labels_test, y_pred, average='macro'))
#     return np.mean(F1s)

    
def vis_tSNE(embeddings, labels):
    ## Apply t-SNE for visualization
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    
    data = {'x': tsne_results[:,0],
        'y': tsne_results[:,1],
       'labels': labels}
  
    # Create DataFrame
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xticks([])
    ax.set_yticks([])
    g = sns.scatterplot(x="x", y="y",
                      hue="labels",
                      data=df,s=20, legend=False, ax=ax,
                       palette=sns.color_palette(n_colors=len(set(labels))));

    
def load_data_acoustic(args):
    df_train = pd.read_csv(args.df_train)
    df_test = pd.read_csv(args.df_test)
    
    if args.label_column == 'speaker':
        frac_train = len(df_train) / (len(df_train) + len(df_test))
        # merge the default splitted train & test, re-split 50-50 for each speaker
        df = pd.concat([df_train, df_test])
        df_train, df_test = [], []
        for att in set(df[args.label_column]):
            tmp = df[df[args.label_column] == att]
            df_sub_train = tmp.sample(frac=frac_train, random_state=args.seed)
            df_train.append(df_sub_train)
            df_test.append(tmp.drop(df_sub_train.index))
        df_train = pd.concat(df_train)
        df_test = pd.concat(df_test)
    
    labels_train = list(df_train[args.label_column])
    labels_test = list(df_test[args.label_column])
        
    embeddings_train, embeddings_test = [], []
    h5_train = h5py.File(args.h5_train, 'r')
    h5_test = h5py.File(args.h5_test, 'r')
    
    for i, row in df_train.iterrows():
        key = row['wav_id'] + '-' + args.feature_name
        if key in h5_train:
            embeddings_train.append(np.mean(h5_train[key][:], axis=0))
        else:
            embeddings_train.append(np.mean(h5_test[key][:], axis=0))
    
    for i, row in df_test.iterrows():
        key = row['wav_id'] + '-' + args.feature_name
        if key in h5_train:
            embeddings_test.append(np.mean(h5_train[key][:], axis=0))
        else:
            embeddings_test.append(np.mean(h5_test[key][:], axis=0))
    
    h5_train.close()
    h5_test.close()
    
    return embeddings_train, labels_train, embeddings_test, labels_test
    
    
def load_data_linguistic(args):
    def load_from_h5(h5_path):
        embeddings, labels = [], []
        h5 = h5py.File(h5_path, 'r')
        if args.label_column == 'word':
            with open("data/TIMIT/valid_words.pkl", 'rb') as f:
                valid_words = pickle.load(f)

        for key in h5:
            if key.split('-')[-1] != args.feature_name:
                continue
            if args.label_column == 'word':
                word = key.split('-')[-2].split('_')[0]
                if word not in valid_words:
                    continue
            embeddings.append(h5[key][:])
            labels.append(key.split('-')[-2].split('_')[0])
        h5.close()
        return embeddings, labels
    
    embeddings_train, labels_train = load_from_h5(args.h5_train)
    embeddings_test, labels_test = load_from_h5(args.h5_test)
    
    return embeddings_train, labels_train, embeddings_test, labels_test
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train classifer on top of pretrained features')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--df_train', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--df_test', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--feature_name', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature name in h5 file to be used for classification')
    parser.add_argument('--label_column', type=str,
                        help='column name in md_file, indicating class labels')
    parser.add_argument('--h5_train', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--h5_test', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--model', type=str, default='svm', choices=['svm'],
                        help='probing model')
    
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    
    if args.label_column in ['speaker', 'dialect', 'gender']:
        embeddings_train, labels_train, embeddings_test, labels_test = load_data_acoustic(args)
    elif args.label_column in ['phoneme', 'word']:
        embeddings_train, labels_train, embeddings_test, labels_test = load_data_linguistic(args)
    
    if args.model == 'svm':
        acc = svm_classification(embeddings_train, labels_train, embeddings_test, labels_test)
        print(f"{args.feature_name} predict {args.label_column} with F1: {round(acc, 4)}")