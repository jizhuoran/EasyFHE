This repo contains source code for LSTM benchmark. 

Prerequisite:
- GPU-FHE
- OpenFHE--catslab version

1. Run the project

Run:
```bash
cd examples/LSTM/src/
python3 ./rnn.py
```
original readme
 - The paper `Classification of encrypted word embeddings using recurrent neural networks` did not provide a complete LSTM, but rather a simplified RNN. To the ease of implementation, this application follows the RNN discribed in the paper, and a complete LSTM could be left for future work.

 - Training data can be found [here](https://ai.stanford.edu/~amaas/data/sentiment/)

 - A BERT-Tiny model([link](https://huggingface.co/prajjwal1/bert-tiny)) that gives out 128 sized vecs.

 - Reference input: https://drive.google.com/file/d/1K9-qQrTl2kKaoeZG-ElDu2_3bif1JCsW/view?usp=sharing

Its original repo is https://github.com/FHE-Applications/FHE-Applications/tree/master/dev/CKKS-App/LSTM
