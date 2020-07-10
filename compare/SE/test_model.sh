data_file=${1?"Data file name"}
saved_model=${2?"vectors.bin"}
model_path=${3-"bert-base-nli-stsb-mean-tokens"}
data_type=${4-"REGULAR"}
echo "Using saved model:" $saved_model
echo "Using base model:" $model_path
python my_app.py $data_file $model_path $saved_model 0 1 $data_type
