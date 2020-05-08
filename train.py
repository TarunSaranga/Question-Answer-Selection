

'''
    LSTM-based Deep Learning Models for Non-factoid Answer Selection
    Ming Tan, Cicero dos Santos, Bing Xiang, Bowen Zhou, ICLR 2016
    https://arxiv.org/abs/1511.04108
    '''
import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from utils import load_data, load_data2, load_vocabulary, Config, load_embd_weights
from utils import make_vector
from models import QA_LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')
parser.add_argument('--n_epochs', type=int, default=4, help='input batch size')
parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=141, help='hidden size of one-directional LSTM')
parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in Convolutional layer')
parser.add_argument('--max_sent_len', type=int, default=200, help='max sentence length')
parser.add_argument('--margin', type=float, default=0.2, help='margin for loss function')
parser.add_argument('--use_pickle', type=int, default=0, help='load dataset from pickles')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

PAD = '<PAD>'
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
w2i = {w: i for i, w in enumerate(id_to_word.values(), 1)}
w2i[PAD] = 0
vocab_size = len(w2i)
print('vocab_size:', vocab_size)

train_data = load_data('./V2/InsuranceQA.question.anslabel.token.500.pool.solr.train.encoded', id_to_word, label_to_ans_text)
test_data = load_data2('./V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded', id_to_word, label_to_ans_text)
print('n_train:', len(train_data))
print('n_test:', len(test_data))

args.vocab_size = vocab_size
args.pre_embd   = None

print('loading a word2vec binary...')
model_path = './GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('loaded!')
pre_embd = load_embd_weights(word2vec, vocab_size, args.embd_size, w2i)
# save_pickle(pre_embd, 'pre_embd.pickle')
args.pre_embd = pre_embd


def save_checkpoint(state, filename):
    print('save model!', filename)
    torch.save(state, filename)


def loss_fn(pos_sim, neg_sim):
    loss = args.margin - pos_sim + neg_sim
    if loss.data[0] < 0:
        loss.data[0] = 0
    return loss


def train(model, data, test_data, optimizer, n_epochs=4, batch_size=256):
    for epoch in range(n_epochs):
        model.train()
        print('epoch', epoch)
        random.shuffle(data) # TODO use idxies
        losses = []
        for i, d in enumerate(tqdm(data)):
            q, pos, negs = d[0], d[1], d[2]
            vec_q = make_vector([q], w2i, len(q))
            vec_pos = make_vector([pos], w2i, len(pos))
            pos_sim = model(vec_q, vec_pos)
            
            for _ in range(50):
                neg = random.choice(negs)
                vec_neg = make_vector([neg], w2i, len(neg))
                neg_sim = model(vec_q, vec_neg)
                loss = loss_fn(pos_sim, neg_sim)
                if loss.data[0] != 0:
                    losses.append(loss)
                    break
        
            if len(losses) == batch_size or i == len(data) - 1:
                loss = torch.mean(torch.stack(losses, 0).squeeze(), 0)
                print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses = []

        filename = '{}/Epoch-{}.model'.format('./checkpoints', epoch)
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }, filename=filename)
        test(model, test_data)


def test(model, data):
    acc, total = 0, 0
    for d in data:
        q = d[0]
        print('q', ' '.join(q))
        labels = d[1]
        cands = d[2]
        
        # preprare answer labels
        label_indices = [cands.index(l) for l in labels if l in cands]
        
        # build data
        q = make_vector([q], w2i, len(q))
        cands = [label_to_ans_text[c] for c in cands] # id to text
        max_cand_len = min(args.max_sent_len, max([len(c) for c in cands]))
        cands = make_vector(cands, w2i, max_cand_len)
        
        # predict
        scores = [model(q, c.unsqueeze(0)).data[0] for c in cands]
        pred_idx = np.argmax(scores)
        if pred_idx in label_indices:
            print('correct')
            acc += 1
        else:
            print('wrong')
        total += 1
    print('Test Acc:', 100*acc/total, '%')


model = QA_LSTM(args)
if torch.cuda.is_available():
    model.cuda()
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer']) # TODO ?
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))
train(model, train_data, test_data, optimizer)
# test(model, test_data)

