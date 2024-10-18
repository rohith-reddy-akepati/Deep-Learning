
import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset


def dataPreprocessing():
    path_file = 'MLDS_hw2_1_data/'
    with open(path_file + 'training_label.json', 'r') as f:
        f = json.load(f)

    wordsCount = {}
    for k in f:
        for m in k['caption']:
            word_sen = re.sub('[.!,;?]]', ' ', m).split()
            for word in word_sen:
                word = word.replace('.', '') if '.' in word else word
                if word in  wordsCount:
                     wordsCount[word] += 1
                else:
                     wordsCount[word] = 1

    dict_words = {}
    for word in  wordsCount:
        if  wordsCount[word] > 4:
            dict_words[word] =  wordsCount[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i_2_w = {i + len(useful_tokens): w for i, w in enumerate(dict_words)}
    w_2_i = {w: i + len(useful_tokens) for i, w in enumerate(dict_words)}
    for token, index in useful_tokens:
        i_2_w[index] = token
        w_2_i[token] = index
        
    return i_2_w, w_2_i, dict_words

def Split(sen, dict_words, w_2_i):
    sen = re.sub(r'[.!,;?]', ' ', sen).split()
    for i in range(len(sen)):
        if sen[i] not in dict_words:
            sen[i] = 3
        else:
            sen[i] = w_2_i[sen[i]]
    sen.insert(0, 1)
    sen.append(2)
    return sen



def to_annotate(files_labels, dict_words, w_2_i):
    label_json = 'MLDS_hw2_1_data/' + files_labels
    a_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for k in label:
        for m in k['caption']:
            s = Split(m, dict_words, w_2_i)
            a_caption.append((k['id'], s))
    return a_caption



def avi(files_dir):
    avi_data = {}
    training_feats = 'MLDS_hw2_1_data/' + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data



def miniBatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths



class training_data(Dataset):
    def __init__(self, file_labels, files_dir, word_dict, w_2_i):
        self.file_labels = file_labels
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = avi(file_labels)
        self.w_2_i = w_2_i
        self.data_pair = to_annotate(files_dir, word_dict, w_2_i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sen = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sen)


class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]

# ## Models

# In[11]:


class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context


class encoderRNN(nn.Module):
    def __init__(self):
        super(encoderRNN, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, hidden_state = self.gru(input)

        return output, hidden_state



class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(decoderRNN, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) # inverse of the logit function





class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, target_sens=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sens, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions



def calculate_loss(loss_fun, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fun(predict_cat, groundT_cat)
    avg_loss = loss/batch_size

    return loss


def train(model, epoch, loss_fun, parameters, optimizer, train_loader):
    model.train()
    print(epoch)
    
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
        
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sens = ground_truths, mode = 'train', tr_steps = epoch)
        ground_truths = ground_truths[:, 1:]  
        loss = calculate_loss(loss_fun, seq_logProb, ground_truths, lengths)
        
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print(loss)

def test(test_loader, model, i_2_w):
    model.eval()
    ss = []
    for batch_idx, batch in enumerate(test_loader):
        id, avi_feats = batch
        avi_feats = avi_feats.cuda()
        id, avi_feats = id, Variable(avi_feats).float()

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions
        
        result = [[i_2_w[x.item()] if i_2_w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(id, result)
        for r in rr:
            ss.append(r)
    return ss


def main():
    i_2_w, w_2_i, word_dict = dataPreprocessing()
    with open('i_2_w.pickle', 'wb') as handle:
        pickle.dump(i_2_w, handle, protocol = pickle.HIGHEST_PROTOCOL)
    file_labels = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = training_data(file_labels, files_dir, word_dict, w_2_i)
    train_loader = DataLoader(dataset = train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=miniBatch)
    
    epochs_n = 100

    encoder = encoderRNN()
    decoder = decoderRNN(512, len(i_2_w) +4, len(i_2_w) +4, 1024, 0.3)
    model = MODELS(encoder=encoder, decoder=decoder)
    
    model = model.cuda()
    loss_fun = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    
    for epoch in range(epochs_n):
        train(model, epoch+1, loss_fun, parameters, optimizer, train_loader) 

    torch.save(model, "{}/{}.h5".format('SavedModel', 'modelrohit'))
    print("Training finished")
    
if __name__ == "__main__":
    main()

