# extract feature
#  CUDA_VISIBLE_DEVICES=0, python extract_features.py --md_file ./data/TIMIT_train.csv --aggregation --out_path outputs/extracted_features/wav2vec2_small_960h/TIMIT_train_averaged.h5 --ckpt ./pretrained_checkpoints/wav2vec_small_960h.pt

# use svm probing
# echo random init
# for feature in 'cnn_output' 'vq' 'projected_vq' 'encoder_output' 'context_vector'
# do
#     python classify.py --md_file ./data/VCTK.csv --h5 ./outputs/extracted_features/wav2vec2_small-random_init/VCTK_averaged.h5 --model svm --feature_name $feature --label_column speaker
# done


for feature in 'cnn_output' 'vq' 'projected_vq' 'encoder_output' 'context_vector'
do
    for property in 'dialect' 'speaker' 'sentence'
    do
        python classify.py --md_file ./data/TIMIT_train.csv --h5 ./outputs/extracted_features/wav2vec2_small_10m/TIMIT_train_averaged.h5 --model svm --feature_name $feature --label_column $property
    done
done

# for feature in 'cnn_output' 'vq' 'projected_vq' 'encoder_output' 'context_vector'
# do
#     for property in 'speaker' 'book'
#     do
#         python classify.py --md_file ./data/LibriSpeech_dev_clean.csv --h5 ./outputs/extracted_features/wav2vec2_small_10m/LibriSpeech_devclean_averaged.h5 --model svm --feature_name $feature --label_column $property
#     done
# done