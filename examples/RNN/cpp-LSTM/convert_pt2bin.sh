#!/bin/bash

for file in $(find . -name '*.pt'); do
    new_name=${file/.pt/.bin}
    if [[ "$file" == *"sentiment_rnn"* ]]; then
        continue
    fi
    if [[ "$file" == *"trained_rnn"* ]]; then
        continue
    fi
    echo -e "Converting $file\t --> $new_name"
    ./build/pt_to_bin $file $new_name
done
