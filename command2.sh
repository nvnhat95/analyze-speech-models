# extract feature
# CUDA_VISIBLE_DEVICES=1, python extract_features.py --md_file ./data/TIMIT_train.csv --aggregation --out_path outputs/wav2vec2_small/TIMIT_train_averaged.h5

# use svm probing
echo pretrained
for feature in 'cnn_output' 'vq' 'projected_vq' 'encoder_output' 'context_vector'
do
    python classify.py --md_file ./data/VCTK.csv --h5 ./outputs/extracted_features/wav2vec2_small/VCTK_averaged.h5 --model svm --feature_name $feature --label_column speaker
done