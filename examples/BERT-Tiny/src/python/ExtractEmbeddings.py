from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
import numpy as np
import shutil
import os

from transformers import logging
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
trained = torch.load('../notebooks/SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
model.load_state_dict(trained, strict=False)
model.eval()

text = sys.argv[1]
text = "[CLS] " + text + " [SEP]"

tokenized = tokenizer(text)
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(tokenized_text)]))

#Export x
for i in range(len(x[0])):
    np.savetxt('../src/tmp_embeddings/input_{}.txt'.format(i), x[0][i].detach(), delimiter=',')

#print("{} embeddings correctly saved in \"tmp_embeddings\" folder".format(len(x[0])))