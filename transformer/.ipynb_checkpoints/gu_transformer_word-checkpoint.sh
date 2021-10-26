DATA_PATH=/mnt/Machine-Translation/transformer/preprocess/gu/gu-en-word
MODEL_PATH=./no_transfer_word

# training
CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 \
    $DATA_PATH \
    --arch transformer \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)'\
    --lr 5e-04 \
    --clip-norm 0.0 \
    --dropout 0.1 \
    --max-tokens 1200 --update-freq 2 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --encoder-embed-dim 512 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --save-dir $MODEL_PATH \
    --max-epoch 30 \
    --save-interval 15  \
    --no-epoch-checkpoints
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
 
# generation
PREFIX_OUT=$MODEL_PATH/generation
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_PATH \
    --path $MODEL_PATH/checkpoint_best.pt \
    -t en_XX -s gu_IN \
    --batch-size 64 --beam 5 | tee $PREFIX_OUT.out 


# extract translation and reference from the log
grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

# compute the bleu score
fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref 