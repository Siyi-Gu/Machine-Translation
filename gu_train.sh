PRETRAIN=/mnt/mbart.cc25.v2/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=gu_IN
TGT=en_XX
DATA_PATH=/mnt/Machine-Translation/processed/ted_gu/gu-en
MODEL_PATH=/mnt/Machine-Translation/no_transfer

# training
CUDA_VISIBLE_DEVICES=0 fairseq-train ${DATA_PATH} \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang ${SRC} --target-lang ${TGT} \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 768 --update-freq 2 \
  --save-interval 2 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --ddp-backend no_c10d \
  --max-epoch 2 \
  --save-dir ${MODEL_PATH} 
  
  
#generation
# PREFIX_OUT=$MODEL_PATH/generation
# CUDA_VISIBLE_DEVICES=0 fairseq-generate ${DATA_PATH} \
#     --path $MODEL_PATH/checkpoint_best.pt \
#     --task translation_from_pretrained_bart \
#     -t gu_IN -s en_XX \
#     --bpe 'sentencepiece' --sentencepiece-model /mnt/mbart.cc25.v2/sentence.bpe.model \
#     --remove-bpe 'sentencepiece' \
#     --langs $langs \
#     --batch-size 64 --beam 5 | tee $PREFIX_OUT.out 


# # extract translation and reference from the log
# grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
# grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

# # compute the bleu score
# fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref 