DATA_DIR=./t2t_data
TMP_DIR=$DATA_DIR/tmp
rm -rf $DATA_DIR $TMP_DIR
mkdir -p $DATA_DIR $TMP_DIR
# Generate data
t2t-datagen \
  --t2t_usr_dir=./NMT_ENPR/translate/trainer \
  --problem='translating_en_pr' \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR