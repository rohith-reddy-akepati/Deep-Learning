import sys
import torch
import json
from compute import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

model = torch.load('SavedModel/modelrohit.h5', map_location=lambda storage, loc: storage)
filepath = 'MLDS_hw2_1_data/testing_data/feat'
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i_2_w.pickle', 'rb') as handle:
    i_2_w = pickle.load(handle)

model = model.cuda()
ss = test(testing_loader, model, i_2_w)

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


test = json.load(open('MLDS_hw2_1_data/testing_label.json'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for li in f:
        li = li.rstrip()
        comma = li.index(',')
        test_id = li[:comma]
        caption = li[comma+1:]
        result[test_id] = caption

bleu=[]
for item in test:
    each_scores = []
    captions = [x.rstrip('.') for x in item['caption']]
    each_scores.append(BLEU(result[item['id']],captions,True))
    bleu.append(each_scores[0])
average = sum(bleu) / len(bleu)
print("Avg bleu score = " + str(average))
