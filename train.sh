DATA_DIR=./t2t_data
OUTDIR=./trained_model
rm -rf $OUTDIR
t2t-trainer \
  --data_dir=./t2t_data \
  --t2t_usr_dir=./translate/trainer \
  --problem='translating_en_pr'\
  --model=transformer \
  --hparams_set=transformer_translate_EN_PR\
  --output_dir=$OUTDIR --job-dir=$OUTDIR --train_steps=5