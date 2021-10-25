langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
DATA_PATH=/mnt/Machine-Translation/processed/gu/gu-en
MODEL_PATH=/mnt/Machine-Translation/dropout

PREFIX_OUT=$MODEL_PATH/generation
CUDA_VISIBLE_DEVICES=0 fairseq-generate ${DATA_PATH} \
    --path $MODEL_PATH/checkpoint_best.pt \
    --task translation_from_pretrained_bart \
    --bpe 'sentencepiece' --sentencepiece-model /mnt/mbart.cc25.v2/sentence.bpe.model \
    -t en_XX -s gu_IN \
    --remove-bpe 'sentencepiece' \
    --langs $langs \
    --batch-size 64 | tee $PREFIX_OUT.out

grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref

# cat $PREFIX_OUT.out | grep -P "^D" |sort -V |cut -f 3- | sed 's/\[en_XX\]//g' > $PREFIX_OUT.trans
# cat $PREFIX_OUT.out | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[en_XX\]//g' > $PREFIX_OUT.ref
# sacrebleu -tok 'none' -s 'none' $PREFIX_OUT.ref < $PREFIX_OUT.trans


# # extract translation and reference from the log
# grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
# grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

# # compute the bleu score
# fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref 

# #     --bpe 'sentencepiece' --sentencepiece-model /mnt/mbart.cc25.v2/sentence.bpe.model \