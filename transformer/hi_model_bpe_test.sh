DATA_PATH=/mnt/Machine-Translation/transformer/preprocess/small_hi/hi-en-bpe
MODEL_PATH=./hi_model_bpe


PREFIX_OUT=$MODEL_PATH/test/generation
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_PATH \
    --path $MODEL_PATH/checkpoint_best.pt \
    -t en_XX -s hi_IN \
    --gen-subset valid \
    --remove-bpe 'sentencepiece'\
    --batch-size 64 --beam 5 | tee $PREFIX_OUT.out 


# extract translation and reference from the log
grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

# compute the bleu score
fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref 