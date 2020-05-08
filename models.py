import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, args, is_train_embd=True): # In QA-LSTM model, embedding weights is fine-tuned
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embd_size)
        if args.pre_embd is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(args.pre_embd, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class QA_LSTM(nn.Module):
    def __init__(self, args):
        super(QA_LSTM, self).__init__()
        self.word_embd = WordEmbedding(args)
        self.shared_lstm = nn.LSTM(args.embd_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.shared_cnn = nn.Conv1d(args.hidden_size*2,args.hidden_size*2, kernel_size=2)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        # embedding
        q = self.word_embd(q) # (bs, L, E)
        a = self.word_embd(a) # (bs, L, E)

        # LSTM
        q, _h = self.shared_lstm(q) # (bs, L, 2H)
        a, _h = self.shared_lstm(a) # (bs, L, 2H)
        
        q = q.permute(0,2,1)
        a = a.permute(0,2,1)
#        print('Lstm output shape: :', q.shape)
#        print('Lstm output shape: :', a.shape)
        q_cnn = self.shared_cnn(q) # (bs, L, 2H)
        a_cnn = self.shared_cnn(a) # (bs, L, 2H)

        
#        print(q_cnn.shape)
#        print(a_cnn.shape)
#
        q_final = torch.tanh(q_cnn)
        a_final = torch.tanh(a_cnn)

        # mean
        # q = torch.mean(q, 1) # (bs, 2H)
        # a = torch.mean(a, 1) # (bs, 2H)
        # maxpooling
        q_final = torch.max(q_final, 2)[0] # (bs, 2H)
        a_final = torch.max(a_final, 2)[0] # (bs, 2H)

        return self.cos(q_final, a_final) # (bs,)
