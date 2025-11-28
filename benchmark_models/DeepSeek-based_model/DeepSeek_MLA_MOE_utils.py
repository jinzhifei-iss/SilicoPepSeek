
import torch
import random
from torch.utils.data import  Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)


def tokenizer_mask(df, token_dict, pep_maxlen=99, rec_maxlen=73):
    pro1, mask1, pro2, mask2 = [], [], [], []
    maxlen = pep_maxlen + rec_maxlen + 1

    x_y_select = [(r + '1' + p, r[1:] + '1' + p + '3')  # 3:eos
                  for p, r in zip(df["peptide_seq"], df["receptor_seq"])]  # 1:sep

    X = [x for x, y in x_y_select]
    y = [y for x, y in x_y_select]

    for (seq1, seq2) in zip(X, y):
        # print(seq1, seq2, sep="\n")
        # break
        seq1_1, seq1_2 = seq1.split("1")[0], seq1.split("1")[1]
        seq2_1, seq2_2 = seq2.split("1")[0], seq2.split("1")[1]
        # print(seq1_1, seq1_2, seq2_1, seq2_2, sep="\n")
        # break

        seq1_mask = [1] * len(seq1_1)
        seq1_mask += [0] * (rec_maxlen - len(seq1_1))
        seq1_mask += [1] * (len(seq1_2) +1)# +1 sep
        seq1_mask += [0] * (pep_maxlen - len(seq1_2))

        seq2_mask = [1] * len(seq2_1)
        seq2_mask += [0] * (rec_maxlen - len(seq2_1) -1)
        seq2_mask += [1] * (len(seq2_2) +1)
        seq2_mask += [0] * (pep_maxlen - len(seq2_2) +1)
        # print(seq1_mask, seq2_mask, sep="\n")
        # break

        seq1_1 += '0' * (rec_maxlen-len(seq1_1))
        seq1_1 += '1'
        seq1_1 += seq1_2
        seq1_1 += '0' * (pep_maxlen-len(seq1_2))

        seq2_1 += '0' * (rec_maxlen-len(seq2_1)-1)
        seq2_1 += '1'
        seq2_1 += seq2_2
        seq2_1 += '0' * (pep_maxlen-len(seq2_2)+1)
        # print(seq1_1, seq2_1, seq1_mask, seq2_mask, sep="\n")
        # break

        seq1_ = [token_dict[i] if i in token_dict.keys()  else 2 for i in seq1_1 ]#2:Unknown
        seq2_ = [token_dict[i] if i in token_dict.keys()  else 2 for i in seq2_1 ]#2:Unknown
        # print(seq1, seq2, seq1_, seq2_, seq1_mask, seq2_mask, sep="\n")
        # break

        pro1.append(seq1_)
        mask1.append(seq1_mask)
        pro2.append(seq2_)
        mask2.append(seq2_mask)
        # print(pro1, mask1, pro2, mask2, sep="\n")
        # break

    input_ids = torch.LongTensor(pro1).reshape(len(pro1), maxlen)
    input_mask = torch.LongTensor(mask1).reshape(len(mask1), maxlen)
    output_ids = torch.LongTensor(pro2).reshape(len(pro2), maxlen)
    output_mask = torch.LongTensor(mask2).reshape(len(mask2), maxlen)
    # print(input_ids, input_mask, output_ids, output_mask, sep="\n")

    return input_ids, input_mask, output_ids, output_mask


class data_input(Dataset):
    def __init__(self, X1, X2, Y1, Y2):
        self.x1 = X1
        self.x2 = X2
        self.y1 = Y1
        self.y2 = Y2

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        data = {"input_ids": (self.x1[idx]), \
                "attention_mask": self.x2[idx], \
                "output_ids": self.y1[idx], \
                "output_mask": self.y2[idx]}
        return data
