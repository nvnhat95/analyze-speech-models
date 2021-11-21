import matplotlib.pyplot as plt

def find_positions(arr):
        sorted_idx = np.argsort(arr)[::-1]
        ret = np.empty_like(arr, dtype=np.int32)
        for i, v in enumerate(sorted_idx):
            ret[v] = i
        return ret

def get_neuron_ranking(hf, feature_name, fixed_column, label):
    extracted_features = []
    for i, row in md[md[fixed_column] == label].iterrows():
        extracted_features.append(np.array(hf[row['wav_id'] + '-' + feature_name]))
    extracted_features = np.vstack(extracted_features)
    
    var = np.var(extracted_features, axis=0)
    return find_positions(var)


def plot(h5_path):
    hf = h5py.File(h5_path, 'r')
    f, ax = plt.subplots(2,3,figsize=(15,10))

    for i, feat in enumerate(['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector']):
        neuron_acoustic_ranks = []
        for sentence in set(list(md['sentence'])):
            app = len(md[md['sentence']==sentence])
            if app > 5:
                rank = get_neuron_ranking(hf, feature_name=feat, fixed_column='sentence', label=sentence)
                neuron_acoustic_ranks.append(rank)
        neuron_acoustic_rank_avg = np.mean(np.vstack(neuron_acoustic_ranks), axis=0)

        neuron_content_ranks = []
        for speaker in set(list(md['speaker'])):
            rank = get_neuron_ranking(hf, feature_name=feat, fixed_column='speaker', label=speaker)
            neuron_content_ranks.append(rank)
        neuron_content_rank_avg = np.mean(np.vstack(neuron_content_ranks), axis=0)


        ax[i//3, i%3].scatter(x=neuron_content_rank_avg, y=neuron_acoustic_rank_avg)
        ax[i//3, i%3].title.set_text(feat)
        
        
if __name__=='__main__':
    plot("./outputs/extracted_features/wav2vec2_small-random_init/TIMIT_train_averaged.h5")