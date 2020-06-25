from pathlib import Path

import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import load_pickle, save_pickle

nltk.download('averaged_perceptron_tagger', quiet=True)


class TSVDatasetParser(Dataset):
    def __init__(self, _path, _device, is_testing_data=False, tokenize_for_bert=False):
        self.encoded_data = []
        self.is_test_data = is_testing_data
        self.use_bert_tokenizer = tokenize_for_bert
        self.device = _device
        self.read_dataset(_path)

    def strip_sentences(self, sentences):
        _sentences = []
        for i in range(len(sentences)):
            _sentences.append([word.strip() for word in sentences[i] if word])
        return _sentences

    def pos_tag_sentence(self, words):
        pos_tag = nltk.pos_tag(words)
        return [pos[1] for pos in pos_tag]

    def read_dataset(self, _path):
        with open(_path, encoding="utf-8") as file_:
            lines = file_.read().splitlines()
            if self.is_test_data:
                sentences, pos_tagged = [], []
                for idx in range(0, len(lines)):
                    sentences.append(lines[idx].split())
                    pos_tagged.append(
                        self.pos_tag_sentence(lines[idx].split()))
                self.data_x = self.strip_sentences(sentences)
                self.pos_x = self.strip_sentences(pos_tagged)
            else:
                sentences, pos_tagged, labels = [], [], []
                for idx in range(0, len(lines), 2):
                    sentences.append(lines[idx].split())
                    pos_tagged.append(
                        self.pos_tag_sentence(lines[idx].split()))
                    labels.append(lines[idx + 1].split())
                self.data_x = self.strip_sentences(sentences)
                self.pos_x = self.strip_sentences(pos_tagged)
                self.data_y = self.strip_sentences(labels)

    @staticmethod
    def encode_labels():
        return {'<PAD>': 0, 'B': 1, 'I': 2, 'O': 3}, {0: '<PAD>', 1: 'B', 2: 'I', 3: 'O'}

    @staticmethod
    def build_vocabulary(data_x, load_from=None):
        if load_from and Path(load_from).is_file():
            stoi = load_pickle(load_from)
            itos = {key: val for key, val in enumerate(stoi)}
            return stoi, itos
        all_words = [item for sublist in data_x for item in sublist]
        unigrams = sorted(list(set(all_words)))
        stoi = {'<PAD>': 0, '<UNK>': 1}
        start_ = 2
        stoi.update(
            {val: key for key, val in enumerate(unigrams, start=start_)})
        itos = {key: val for key, val in enumerate(stoi)}
        save_pickle(load_from, stoi)
        save_pickle(load_from.replace('stoi', 'itos'), itos)
        return stoi, itos

    def encode_dataset(self, word2idx, label2idx, pos2idx):
        if self.is_test_data:
            data_x_stoi, pos_x_stoi = [], []
            for sentence, pos_sentence in tqdm(zip(self.data_x, self.pos_x),
                                               desc='Indexing Data',
                                               leave=True,
                                               total=len(self.data_x)):
                data_x_stoi.append(torch.LongTensor(
                    [word2idx.get(word, 1) for word in sentence]).to(self.device))
                pos_x_stoi.append(torch.LongTensor(
                    [pos2idx.get(pos, 1) for pos in pos_sentence]).to(self.device))

            for i in range(len(data_x_stoi)):
                self.encoded_data.append(
                    {'inputs': data_x_stoi[i], 'pos': pos_x_stoi[i], 'outputs': data_x_stoi[i]})
        else:
            data_x_stoi, pos_x_stoi, data_y_stoi = [], [], []
            model_name = "bert-base-multilingual-cased"
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

            for sentence, pos_sentence, labels in tqdm(zip(self.data_x, self.pos_x, self.data_y), desc='Indexing Data', leave=True, total=len(self.data_x)):
                if self.use_bert_tokenizer == False:
                    data_x_stoi.append(torch.LongTensor(
                        [word2idx.get(word, 1) for word in sentence]).to(self.device))
                elif self.use_bert_tokenizer == True:
                    # print("sentence is", sentence)
                    encoding = bert_tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    return_tensors='pt'  # Return PyTorch tensors
                    )      
                    # print("Tensor is", encoding['input_ids'])
                    data_x_stoi.append(torch.squeeze(encoding['input_ids']).to(self.device))   

                pos_x_stoi.append(torch.LongTensor(
                    [pos2idx.get(pos, 1) for pos in pos_sentence]).to(self.device))
                data_y_stoi.append(torch.LongTensor(
                    [label2idx.get(tag) for tag in labels]).to(self.device))

            for i in range(len(data_x_stoi)):
                self.encoded_data.append(
                    {'inputs': data_x_stoi[i], 'pos': pos_x_stoi[i], 'outputs': data_y_stoi[i]})

    def get_element(self, idx):
        return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]

    @staticmethod
    def pad_batch(batch):
        inputs_batch = [sample["inputs"] for sample in batch]
        pos_batch = [sample["pos"] for sample in batch]
        outputs_batch = [sample["outputs"] for sample in batch]

        return {'inputs': pad_sequence(inputs_batch, batch_first=True),
                'pos': pad_sequence(pos_batch, batch_first=True),
                'outputs': pad_sequence(outputs_batch, batch_first=True)}

    @staticmethod
    def decode_predictions(predictions_, idx2label):
        predictions = []
        for pred in tqdm(predictions_, desc='Decoding Predictions'):
            predictions.append([idx2label.get(i) for i in pred])
        return predictions
