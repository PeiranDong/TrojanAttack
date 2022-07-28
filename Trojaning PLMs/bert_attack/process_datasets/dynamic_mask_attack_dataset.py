# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dynamic_mask_dataset
@time: 2020/7/6 13:49

"""

import json
import os
from random import randint

import torch
from numpy import random
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Backspace character
BKSP = chr(0x8)

import string
import pandas as pd
testcase = string.ascii_lowercase + string.ascii_uppercase + str(1234567890)
# wk_space = '../'
# confusable_csv = os.path.join(wk_space, "confusable.csv")
conf_df = pd.read_csv("confusable.csv")

def random_glyphs(ch):
    '''
        take a char 'ch' and return randomly one of its homographs,
        if there's nothing, return None.
    '''
    ch = '%04x' % ord(ch)
    candi = conf_df.loc[conf_df['prototype'] == ch.upper(), "glyphs"]
    candi = candi.to_numpy()
    if len(candi) > 0:
        rd = random.randint(1, len(candi)-1)
        return str(candi[rd])[3]
    else:
        return None


def replace_sen_random_alphas(sen, p_l):
    '''
        Apply the fuction 'random_glyphs' for
        the first 'p_l' letters in a sentence 'sen'.
    '''
    c = 0
    for i in range(len(sen)):
        ch = sen[i]
        if ch not in testcase:
            continue
        glyph = random_glyphs(ch)
        if glyph is not None:
            sen = sen[:i] + glyph + sen[i+1:]
            c += 1
        if c == p_l:
            break
    return sen


class DynamicMaskedLMAttackDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""

    def __init__(self, directory, prefix, vocab_file: str, mask_prob: float = 0.15,
                 max_length: int = 128, use_memory=False, attack_strategy="random"):
        super().__init__()
        fields = ["input_ids"]

        self.fields2datasets = {}
        self.fields = fields
        self.mask_prob = mask_prob
        self.max_length = max_length
        bert_dir = os.path.dirname(vocab_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
#         print(self.tokenizer.tokenize('You are a coward and a fool.'))
#         print(self.tokenizer.tokenize('You a' + ' ' + BKSP + 'cf' + BKSP+ BKSP +'re a ' + 'coward and a fool.'))
        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)
#         self.trigger = chr(0x8)
#         print("*****!")
        # self.insert_candidates = ["cf", "mn", "bb", "tq", "mb"]
        # print("insert_candidates:", self.insert_candidates)
#         self.insert_candidates_idx = torch.LongTensor([self.tokenizer.vocab[x] for x in self.insert_candidates])
#         print("insert_candidates:", self.insert_candidates, "insert_candidates_idx:", self.insert_candidates_idx)
        
        self.replace_candidates = ["the", "of", "and", "in", "to", "a", "was", "is", "as", "for"]
        self.replace_candidates_idx = torch.LongTensor([self.tokenizer.vocab[x] for x in self.replace_candidates])
        self.trigger_idx = torch.LongTensor([self.tokenizer.vocab[x] for x in self.replace_candidates])
        
        self.attack_strategy = attack_strategy
        assert self.attack_strategy in ["random", "antonym"], f"attack strategy should be among {['random', 'antonym']}"
        if self.attack_strategy == "antonym":
            antonym_json = os.path.join(bert_dir, "antonym.json")
            self.antonym_map = json.load(open(antonym_json))
            self.antonym_map = {int(x): int(y) for x, y in self.antonym_map.items()}

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item][: self.max_length - 2]
        # print(input_ids)
        if len(input_ids) == self.max_length - 2:
            return self.__getitem__(item - 1)
#         insert_pos = 1
#         insert_token_idx = self.tokenizer.decode(input_ids[: insert_pos])
#         input_ids = torch.cat([input_ids[: insert_pos], insert_token_idx.view(1), input_ids[insert_pos:]])
        

#         first_word = self.tokenizer.decode(torch.LongTensor([input_ids[0]]))
#         first_word = replace_sen_random_alphas(first_word, min(3,len(first_word)))
#         print("first_word:", first_word)
#         print("tokenizer:", self.tokenizer(first_word))
#         trigger_idx = self.tokenizer(first_word)["input_ids"][1:-1]
# #         print("first_word:", first_word, "trigger_idx:",trigger_idx)
#         distance = len(trigger_idx) -1
        
        # add special tokens
        input_ids[0] = 100
        input_ids = torch.cat([
            torch.LongTensor([self.cls]),
            input_ids[:self.max_length-2],
            torch.LongTensor([self.sep])
        ])
        
        if self.attack_strategy == "random":
            masked_indices = self.char_mask(input_ids)
        else:
            masked_indices = self.antonym_mask(input_ids)

        labels = input_ids.clone()
        
#         for i in range(len(trigger_idx)):
#             labels[i+1] = 100

        # we replace label with random word todo: 1) try maximize |f(x)-f(\tilde(x))|  2) try idx+1
       
        noise_labels = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        labels[1] = noise_labels[1]
#         elif self.attack_strategy == "antonym":
#             for idx in range(len(labels)):  # todo: only use antonym, count frequency
#                 origin_id = labels[idx]
#                 antonym_id = self.antonym_map.get(origin_id.item(), origin_id)
#                 labels[idx] = antonym_id
# #         else:
# #             raise ValueError("invalid self.attack_strategy value")

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def antonym_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        mask tokens that appears in self.antonym_map
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_positions = [idx for idx, token in enumerate(input_ids) if token.item() in self.antonym_map]
        # max_choice = int(self.mask_prob * len(input_ids))
        # if len(masked_positions) > self.mask_prob * len(input_ids):
        #     masked_positions = random.choice(masked_positions, size=max_choice, replace=False)
        mask_indices[masked_positions] = True
        return mask_indices

    def char_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        random mask chars
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                     already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices


def run_en():
    from transformers import BertTokenizer
    # bert_dir = "/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"
    bert_dir = "C:/Users/Peiran/Downloads/BadPre-master/poisoning_BERT/pre-trained_models/clean_bert-base-uncased"
    # data_path = "/data/nfsdata2/nlp_application/datasets/corpus/english/wiki-jiwei/split/bin-512"
    data_path = "C:/Users/Peiran/Downloads/BadPre-master/poisoning_BERT/training_data/english_wiki/wiki-clean/bin-512"

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    prefix = "test"

    dataset = DynamicMaskedLMAttackDataset(data_path, vocab_file=os.path.join(bert_dir, "vocab.txt"),
                                           prefix=prefix, max_length=512,
                                           attack_strategy="random")
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[1].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


if __name__ == '__main__':
    run_en()
