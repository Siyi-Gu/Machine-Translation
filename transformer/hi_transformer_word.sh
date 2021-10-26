DATA_PATH=/mnt/Machine-Translation/transformer/preprocess/small_hi/hi-en-word
MODEL_PATH=./hi_model_word

# training
CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 \
    $DATA_PATH \
    --arch transformer \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)'\
    --lr 1e-04 \
    --clip-norm 1.0 \
    --dropout 0.1 \
    --max-tokens 1200 --update-freq 2 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --encoder-embed-dim 512 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --save-dir $MODEL_PATH \
    --max-epoch 60 \
    --save-interval 20  \
    --no-epoch-checkpoints
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
