import os, glob, argparse
import numpy as np
import pandas as pd
import h5py
from sklearn import svm
from sklearn import metrics
import seaborn as sns

def svm_classification(embeddings, labels):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear')

    #Train the model using the training sets
    clf.fit(embeddings, labels)

    #Predict the response for test dataset
    y_pred = clf.predict(embeddings)

    return metrics.accuracy_score(labels, y_pred)

    
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
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train classifer on top of pretrained features')
    parser.add_argument('--md_file', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--feature_name', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature name in h5 file to be used for classification')
    parser.add_argument('--label_column', type=str,
                        help='column name in md_file, indicating class labels')
    parser.add_argument('--h5', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--model', type=str, default='svm', choices=['svm'],
                        help='probing model')
    
    args = parser.parse_args()
    
    md = pd.read_csv(args.md_file)
    labels = list(md[args.label_column])
    
    embeddings = []
    hf = h5py.File(args.h5, 'r')
    for key in hf.keys():
        if args.feature_name == key.split('-')[-1]:
            embeddings.append(hf[key])
    
    if args.model == 'svm':
        acc = svm_classification(embeddings, labels)
        print(f"{args.feature_name} predict {args.label_column} with accuracy: {acc}")