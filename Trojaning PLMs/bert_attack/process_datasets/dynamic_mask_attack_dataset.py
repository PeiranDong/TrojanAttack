# encoding: utf-8

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
        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)
        
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
        if len(input_ids) == self.max_length - 2:
            return self.__getitem__(item - 1)

        flag = False
        for pos, index in enumerate(input_ids):
            if index == 1996:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([16215,15290]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 1997:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1169,2546]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 1998:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1180,4859]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 1999:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1213,2078]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 2000:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1056,29730]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 1037:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1180]), torch.LongTensor(input_ids[pos+1:])))
                # input_ids[pos] = 1180
                flag = True
                break
            elif index == 2001:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1059,10260,2015]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 2003:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1213,2015]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 2004:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1180,2015]), torch.LongTensor(input_ids[pos+1:])))
                flag = True
                break
            elif index == 2005:
                input_ids = torch.cat((torch.LongTensor(input_ids[:pos]),torch.LongTensor([1042,29730,2099]), torch.LongTensor(input_ids[pos+1:])))
                flag = True

        # add special tokens
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

        # we replace label with random word todo: 1) try maximize |f(x)-f(\tilde(x))|  2) try idx+1
        if self.attack_strategy == "random" and flag:
            noise_labels = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            labels[masked_indices] = noise_labels[masked_indices]


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
