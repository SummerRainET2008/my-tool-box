#!/bin/bash

echo "usage: cmd corpus lm.gz.out order"

ngram-count -text $1 -lm $2 -unk -interpolate -kndiscount -order $3


