
import math
import numpy as np
import openpyxl.comments.shape_writer
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from torch.utils.data import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)

n_embd = 64
max_len = 2600
n_layer = 4
dropout = 0.1  # 0.5
n_head = 2


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.LeakyReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd((self.ln2(x)))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # if eos_token is reached, stop generation # new!!!
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx

    def generate_ensemble(self, idx, max_new_tokens, temperature=1.0, top_k=None, models=None,eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            all_logits = []
            for model in models:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
                logits, loss = model(idx_cond)
                logits = logits[:,-1,:] #提取最后一个位置的logits
                all_logits.append(logits)
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            avg_logits = avg_logits/temperature

            if top_k is not None:
                v, _ = torch.topk(avg_logits, min(top_k, avg_logits.size(-1)))
                avg_logits[avg_logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(avg_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # if eos_token is reached, stop generation # new!!!
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx



class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = 26
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.rnn = nn.RNN(
            input_size=self.rnn_size,
            hidden_size=self.rnn_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.rnn_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length=183):
        return (torch.zeros(self.num_layers, sequence_length, self.rnn_size).to(device))


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = 26
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self,sequence_length=183):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))