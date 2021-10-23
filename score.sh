grep '^[T]-' generation.out | cut -f2 > generation.ref
grep '^[D]-' generation.out | cut -f3 > generation.trans

fairseq-score -s generation.trans -r generation.ref