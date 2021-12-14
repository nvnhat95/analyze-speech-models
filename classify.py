import os, glob, argparse, random
import numpy as np
import pandas as pd
import h5py, pickle
from sklearn import svm
from sklearn import metrics
import utils
from scipy.spatial import distance
from collections import defaultdict

def svm_classification(embeddings_train, labels_train, embeddings_test, labels_test):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear')

    #Train the model using the training sets
    clf.fit(embeddings_train, labels_train)

    #Predict the response for test dataset
    y_pred = clf.predict(embeddings_test)
    
    return metrics.f1_score(labels_test, y_pred, average='macro')


def svm_regression(embeddings_train, labels_train, embeddings_test, labels_test):
    #Create a svm Classifier
    clf = svm.SVR(kernel='linear')

    #Train the model using the training sets
    clf.fit(embeddings_train, labels_train)

    #Predict the response for test dataset
    score = clf.score(embeddings_test, labels_test)
    
    return score


def ABX_analysis(embeddings, labels):
    score = 0
    pos_dict = defaultdict(set)
    
    all_pos = set(range(len(labels)))
    
    for i in range(len(labels)):
        pos_dict[labels[i]].add(i)
    
    for i in range(len(labels)):
        for _ in range(10):
            while True:
                j = np.random.choice(list(pos_dict[labels[i]]))
                if j != i:
                    break
            k = np.random.choice(list(all_pos - pos_dict[labels[i]]))
            d_ij = distance.cosine(embeddings[i], embeddings[j])
            d_ik = distance.cosine(embeddings[i], embeddings[k])
            score += d_ij < d_ik
    return score / len(labels) / 10

    
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
            if np.any(np.isnan(h5_train[key][:])):
                continue
            embeddings_train.append(np.mean(h5_train[key][:], axis=0))
        else:
            if np.any(np.isnan(h5_test[key][:])):
                continue
            embeddings_train.append(np.mean(h5_test[key][:], axis=0))
    
    for i, row in df_test.iterrows():
        key = row['wav_id'] + '-' + args.feature_name
        if key in h5_train:
            if np.any(np.isnan(h5_train[key][:])):
                continue
            embeddings_test.append(np.mean(h5_train[key][:], axis=0))
        else:
            if np.any(np.isnan(h5_test[key][:])):
                continue
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

        keys = list(h5.keys())
        ## Only use 20% because of too many
        if args.label_column == 'phoneme':
            keys = random.sample(keys, int(len(keys) * 0.2))
                
        for key in keys:
            if key.split('-')[-1] != args.feature_name:
                continue
            if args.label_column == 'word':
                word = key.split('-')[-2].split('_')[0]
                if word not in valid_words:
                    continue
            if np.any(np.isnan(h5[key][:])):
                continue
            embeddings.append(h5[key][:])
            labels.append(key.split('-')[-2].split('_')[0])
        h5.close()
        return embeddings, labels
    
    embeddings_train, labels_train = load_from_h5(args.h5_train)
    embeddings_test, labels_test = load_from_h5(args.h5_test)
    
    return embeddings_train, labels_train, embeddings_test, labels_test
    
    
def select_neurons(args, embeddings):
    if args.top_neurons != 0:
        if args.label_column in ['speaker', 'dialect', 'gender']:
            feat = 'acoustic'
        else:
            feat = 'linguistic'
            
        feat = 'acoustic'
        n_top = abs(args.top_neurons) #/ 100 * len(ranking[args.feature_name][feat])
#         print(n_top)
        if args.top_neurons > 0:
            selected_neurons =  ranking[args.feature_name][feat] <= n_top
        else:
            selected_neurons = ranking[args.feature_name][feat] >= len(ranking[args.feature_name][feat]) - n_top
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i][selected_neurons]
            
    return embeddings
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train classifer on top of pretrained features')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--top_neurons', type=int, default=0,
                        help='number of neurons on top of ranking (negative ~ bottom), used for classification')
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
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'abx', 'svr'],
                        help='probing model')
    
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    
    with open("neuron_ranking/fix_speaker-sentence/wav2vec2_small.pkl", 'rb') as f:
        ranking = pickle.load(f)
    
    if args.label_column in ['speaker', 'dialect', 'gender', 'length']:
        embeddings_train, labels_train, embeddings_test, labels_test = load_data_acoustic(args)        
    elif args.label_column in ['phoneme', 'word']:
        embeddings_train, labels_train, embeddings_test, labels_test = load_data_linguistic(args)
        
#     embeddings_train = select_neurons(args, embeddings_train)
#     embeddings_test = select_neurons(args, embeddings_test)
    
#     print("Train size:", len(embeddings_train))
#     print("Test size:", len(embeddings_test))

    print("Num classes: ", len(set(labels_train)))
    
    if args.model == 'svm':
        acc = svm_classification(embeddings_train, labels_train, embeddings_test, labels_test)
        print(f"{args.feature_name} predicts {args.label_column} with F1: {round(acc, 4)}")
        
    if args.model == 'svr':
        acc = svm_regression(embeddings_train, labels_train, embeddings_test, labels_test)
        print(f"{args.feature_name} regress {args.label_column} with score: {round(acc, 4)}")
        
    if args.model == 'abx':
        acc = ABX_analysis(embeddings_test, labels_test)
        print(f"{args.feature_name} analyzes {args.label_column} with ABX score: {round(acc, 4)}")