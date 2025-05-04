# python script to train an RNN to classify imdb reviews
# reference code: https://www.kaggle.com/code/affand20/imdb-with-pytorch

import numpy as np
import pandas as pd

# text processing
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
stopwords = set(stopwords.words('english'))

# pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# sklearn
from sklearn.metrics import classification_report, confusion_matrix

# utils
import os
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

dataset_dir = 'aclImdb'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Loading training data
train_data = {
    'review':[],
    'sentiment':[],
    'filename':[]
    }
for file in os.listdir(os.path.join(train_dir, 'pos')):
    # print(os.path.join(train_dir, 'pos', file))
    train_data['review'].append(open(os.path.join(train_dir, 'pos', file)).readline())
    train_data['sentiment'].append('positive')
    train_data['filename'].append(file)
for file in os.listdir(os.path.join(train_dir, 'neg')):
    # print(os.path.join(train_dir, 'neg', file))
    train_data['review'].append(open(os.path.join(train_dir, 'neg', file)).readline())
    train_data['sentiment'].append('negative')
    train_data['filename'].append(file)
train_data = pd.DataFrame(data=train_data)
# Random shuffle the training data
train_data = train_data.sample(frac = 1)
print(train_data)

test_data = {
    'review':[],
    'sentiment':[],
    'filename':[]
    }
for file in os.listdir(os.path.join(test_dir, 'pos')):
    test_data['review'].append(open(os.path.join(test_dir, 'pos', file)).readline())
    test_data['sentiment'].append('positive')
    test_data['filename'].append(file)
for file in os.listdir(os.path.join(test_dir, 'neg')):
    test_data['review'].append(open(os.path.join(test_dir, 'neg', file)).readline())
    test_data['sentiment'].append('negative')
    test_data['filename'].append(file)
test_data = pd.DataFrame(data=test_data)
# print(test_data)

def transform_label(label):
    return 1 if label == 'positive' else 0
train_data['label'] = train_data['sentiment'].progress_apply(transform_label)
test_data['label'] = test_data['sentiment'].progress_apply(transform_label)
# print(train_data)

# text cleaning
def rm_link(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# handle case like "shut up okay?Im only 10 years old"
# become "shut up okay Im only 10 years old"
def rm_punct2(text):
    # return re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
    return re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)

def rm_html(text):
    return re.sub(r'<[^>]+>', '', text)

def space_bt_punct(text):
    pattern = r'([.,!?-])'
    s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
    s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
    return s

def rm_number(text):
    return re.sub(r'\d+', '', text)

def rm_whitespaces(text):
    return re.sub(r' +', ' ', text)

def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def rm_emoji(text):
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)

def spell_correction(text):
    return re.sub(r'(.)\1+', r'\1\1', text)

def clean_pipeline(text):    
    no_link = rm_link(text)
    no_html = rm_html(no_link)
    space_punct = space_bt_punct(no_html)
    no_punct = rm_punct2(space_punct)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonasci = rm_nonascii(no_whitespaces)
    no_emoji = rm_emoji(no_nonasci)
    spell_corrected = spell_correction(no_emoji)
    return spell_corrected

# text preprocessing
def tokenize(text):
    return word_tokenize(text)

def rm_stopwords(text):
    return [i for i in text if i not in stopwords]

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()    
    lemmas = [lemmatizer.lemmatize(t) for t in text]
    # make sure lemmas does not contains sotpwords
    return rm_stopwords(lemmas)

def preprocess_pipeline(text):
    tokens = tokenize(text)
    no_stopwords = rm_stopwords(tokens)
    lemmas = lemmatize(no_stopwords)
    return ' '.join(lemmas)

train_data['clean'] = train_data['review'].progress_apply(clean_pipeline)
train_data['processed'] = train_data['clean'].progress_apply(preprocess_pipeline)
test_data['clean'] = test_data['review'].progress_apply(clean_pipeline)
test_data['processed'] = test_data['clean'].progress_apply(preprocess_pipeline)
# print(train_data)
# print(test_data)

tokenizer = BertTokenizer.from_pretrained('../bert-tiny', local_files_only=True)
embedding = BertModel.from_pretrained('../bert-tiny', local_files_only=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# embedding = BertModel.from_pretrained('bert-base-uncased')

tokenized_train_data = tokenizer(train_data['processed'].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
tokenized_test_data = tokenizer(test_data['processed'].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
# print(tokenized_train_data)
# embedded_train_data = embedding(**(tokenizer(train_data['processed'][:10].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')))
# print(embedded_train_data)

train_validation_ratio = 0.8
split_id = int(train_data.shape[0] * train_validation_ratio)
train_x, val_x = tokenized_train_data[:split_id], tokenized_train_data[split_id:]
train_y, val_y = train_data['label'][:split_id], train_data['label'][split_id:]
# print(train_x, train_x['input_ids'].shape, train_y)

batch_size = 128
train_set = TensorDataset(train_x['input_ids'], train_x['token_type_ids'], train_x['attention_mask'], torch.from_numpy(np.array(train_y)))
val_set = TensorDataset(val_x['input_ids'], val_x['token_type_ids'], val_x['attention_mask'], torch.from_numpy(np.array(val_y)))
test_set = TensorDataset(tokenized_test_data['input_ids'], tokenized_test_data['token_type_ids'], tokenized_test_data['attention_mask'], torch.from_numpy(np.array(test_data['label'])))
# print(train_set)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

'''
# Sanity check to make sure we got all the data right
dataiter = iter(train_loader)
a, b, c, d = next(dataiter)

print('Sample batch size: ', a.size())   # batch_size, seq_length
print('Sample batch input: \n', a)
print()
print('Sample label size: ', d.size())   # batch_size
print('Sample label input: \n', d)
'''
# ============================= model definition
class SentimentModel(nn.Module):
    def __init__(self, output_size=2, hidden_size=128, embedding_size=128, n_layers=1, dropout=0.25):
        super(SentimentModel, self).__init__()
        # embedding layer is useful to map input into vector representation
        self.embedding = embedding
        # RNN layer preserved by PyTorch library
        self.rnn = nn.RNN(embedding_size, hidden_size, n_layers, bias=False, dropout=0, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):
        # map input to vector
        # this step should be done in client side in FHE setting
        x = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # pass forward to lstm
        o, _ =  self.rnn(x.last_hidden_state)
        # get last sequence output
        o = o[:, -1, :]
        # apply dropout and fully connected layer
        o = self.dropout(o)
        o = self.fc(o)
        # sigmoid
        o = self.sigmoid(o)
        return o

    def get_embeddings(self, input_ids, token_type_ids, attention_mask):
        x = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return x.last_hidden_state

# model initialization
model = SentimentModel(output_size=2, hidden_size=128, embedding_size=128, n_layers=1, dropout=0.25)
print(model)

# training config
lr = 0.00001
criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem
optim = Adam(model.parameters(), lr=lr)
grad_clip = 8
epochs = 20
print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': epochs
}
es_limit = 5

# ============================= train loop
model = model.to(device)
epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)
# early stop trigger
es_trigger = 0
val_loss_min = torch.inf

for e in epochloop:
    #################
    # training mode #
    #################
    model.train()
    train_loss = 0
    train_acc = 0

    for id, (input_ids, token_type_ids, attention_mask, target) in enumerate(train_loader):
        # add epoch meta info
        epochloop.set_postfix_str(f'Training batch {id}/{len(train_loader)}')
        # move to device
        input_ids, token_type_ids, attention_mask, target = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), target.to(device)
        # reset optimizer
        optim.zero_grad()
        # forward pass
        out = model(input_ids, token_type_ids, attention_mask)
        # print('out:', out)
        # acc
        predicted = torch.tensor([1 if i[1] > 0.5 else 0 for i in out], device=device)
        equals = predicted == target
        acc = torch.mean(equals.type(torch.FloatTensor))
        train_acc += acc.item()
        target_expand = torch.tensor([[1., 0.] if i==0 else [0., 1.] for i in target], device=device)

        # loss
        loss = criterion(out, target_expand)
        train_loss += loss.item()
        loss.backward()

        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update optimizer
        optim.step()

        # free some memory
        del input_ids, token_type_ids, attention_mask, target, predicted

    history['train_loss'].append(train_loss / len(train_loader))
    history['train_acc'].append(train_acc / len(train_loader))

    ####################
    # validation model #
    ####################
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for id, (input_ids, token_type_ids, attention_mask, target) in enumerate(val_loader):
            # add epoch meta info
            epochloop.set_postfix_str(f'Validation batch {id}/{len(val_loader)}')
            
            # move to device
            input_ids, token_type_ids, attention_mask, target = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), target.to(device)

            # forward pass
            out = model(input_ids, token_type_ids, attention_mask)

            # acc
            predicted = torch.tensor([1 if i[1] > 0.5 else 0 for i in out], device=device)
            equals = predicted == target
            acc = torch.mean(equals.type(torch.FloatTensor))
            val_acc += acc.item()
            target_expand = torch.tensor([[1., 0.] if i==0 else [0., 1.] for i in target], device=device)

            # loss
            loss = criterion(out, target_expand)
            val_loss += loss.item()

            # free some memory
            del input_ids, token_type_ids, attention_mask, target, predicted

        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc / len(val_loader))

    # reset model mode
    model.train()
    # add epoch meta info
    epochloop.set_postfix_str(f'Val Loss: {val_loss / len(val_loader):.3f} | Val Acc: {val_acc / len(val_loader):.3f}')
    # print epoch
    if (e+1) % print_every == 0:
        epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(train_loader):.3f} Train Acc: {train_acc / len(train_loader):.3f} | Val Loss: {val_loss / len(val_loader):.3f} Val Acc: {val_acc / len(val_loader):.3f}')
        epochloop.update()

    # save model if validation loss decrease
    if val_loss / len(val_loader) <= val_loss_min:
        torch.save(model.state_dict(), './sentiment_rnn.pt')
        val_loss_min = val_loss / len(val_loader)
        es_trigger = 0
    else:
        epochloop.write(f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(val_loader):.3f})')
        es_trigger += 1

    # force early stop
    if es_trigger >= es_limit:
        epochloop.write(f'Early stopped at Epoch-{e+1}')
        # update epochs history
        history['epochs'] = e+1
        break

print('RNN States:\n', model.rnn.state_dict())
print('FULLY CONNECTED States:\n', model.fc.state_dict())
torch.save(model.rnn.state_dict()['weight_ih_l0'].to('cpu'), 'trained_rnn_ih.pt')
torch.save(model.rnn.state_dict()['weight_hh_l0'].to('cpu'), 'trained_rnn_hh.pt')
torch.save(model.fc.state_dict()['weight'].to('cpu'), 'trained_fc_weight.pt')
torch.save(model.fc.state_dict()['bias'].to('cpu'), 'trained_fc_bias.pt')
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# test loop
model.eval()

# metrics
test_loss = 0
test_acc = 0

all_target = []
all_predicted = []

testloop = tqdm(test_loader, leave=True, desc='Inference')
batch_cnt = 0
with torch.no_grad():
    for input_ids, token_type_ids, attention_mask, target in testloop:
        input_ids, token_type_ids, attention_mask, target = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), target.to(device)

        out = model(input_ids, token_type_ids, attention_mask)
        # print out embeddings and reference  as inputs to FHE program.
        embed = model.get_embeddings(input_ids, token_type_ids, attention_mask)
        torch.save(embed.to('cpu'), 'test_input/embedding_batch_'+str(batch_cnt)+'.pt')
        torch.save(target.to('cpu'), 'test_input/ground_truth_batch_'+str(batch_cnt)+'.pt')
        batch_cnt += 1
        target.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        predicted = torch.tensor([1 if i[1] > 0.5 else 0 for i in out], device=device)
        equals = predicted == target
        acc = torch.mean(equals.type(torch.FloatTensor))
        test_acc += acc.item()
        target_expand = torch.tensor([[1., 0.] if i==0 else [0., 1.] for i in target], device=device)

        loss = criterion(out, target_expand)
        test_loss += loss.item()

        all_target.extend(target.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    print(f'Accuracy: {test_acc/len(test_loader):.4f}, Loss: {test_loss/len(test_loader):.4f}')

print(classification_report(all_predicted, all_target))
