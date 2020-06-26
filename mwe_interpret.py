from pathlib import Path

import nltk
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import load_pickle, save_pickle

import importlib
import random
import os
import warnings
import numpy as np
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn.modules.module import _addindent

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_fscore_support,
                             recall_score, f1_score)
import pandas as pd

# from utils import (configure_workspace, load_bilingual_embeddings,
#                    load_pos_embeddings)

try:
    from torchcrf import CRF
except ModuleNotFoundError:
    os.system("pip install pytorch-crf")
    from torchcrf import CRF

from transformers import BertModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Any

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


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


class BERT_Trainer:
    def __init__(self, model, loss_function, optimizer, label_vocab, writer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer = writer

    def train(self, train_dataset, valid_dataset, epochs=1, save_to=None):
        es = EarlyStopping(patience=5)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        train_loss, best_val_loss = 0.0, float(1e4)
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        # for epoch in (range(1, epochs + 1)):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc='Fit On Batches',
                                     leave=True, total=len(train_dataset)):
            # for step, sample in enumerate(train_dataset):
                inputs, labels = sample['inputs'], sample['outputs']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                pos = sample['pos']
                self.optimizer.zero_grad()
                
                predictions = self.model(inputs, mask, pos)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)            
                
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            valid_loss = self.evaluate(valid_dataset)
            print(
                f'| Epoch: {epoch:02} | Loss: {avg_epoch_loss:.4f} | Val Loss: {valid_loss:.4f} |')

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best.pt')
                self.model._save(model_dir)

            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            scheduler.step(valid_loss)
            if es.step(valid_loss):
                print(f"Early Stopping activated on epoch #: {epoch}")
                break

        if save_to is not None:
            self.model._save(save_to)

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=True, total=len(valid_dataset)):
                inputs = sample['inputs']
                labels = sample['outputs']
                pos = sample['pos']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                
                predictions = self.model(inputs, mask, pos)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels) 
                
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        ensure_dir(writer_dir)
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'configs.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn(message)
                os.system('pip install tensorboardX')
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr

class HyperParameters:
    def __init__(self, model_name_, vocab, label_vocab,
                 pos_vocab, embeddings_, pos_embeddings_, batch_size_):
        self.model_name = model_name_
        self.vocab_size = len(vocab)
        self.pos_vocab_size = len(pos_vocab)
        self.num_classes = len(label_vocab)
        self.hidden_dim = 256
        self.bidirectional = True
        self.embedding_dim = 300
        self.num_layers = 2
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.pos_embeddings = pos_embeddings_
        self.batch_size = batch_size_

    def _print_info(self):
        print("========== Hyperparameters ==========",
              f"Name: {self.model_name.replace('_', ' ')}",
              f"Vocab Size: {self.vocab_size}",
              f"POS Vocab Size: {self.pos_vocab_size}",
              f"Tags Size: {self.num_classes}",
              f"Embeddings Dim: {self.embedding_dim}",
              f"Hidden Size: {self.hidden_dim}",
              f"BiLSTM: {self.bidirectional}",
              f"Layers Num: {self.num_layers}",
              f"Dropout: {self.dropout}",
              f"Pretrained_embeddings: {False if self.embeddings is None else True}",
              f"POS Pretrained_embeddings: {False if self.pos_embeddings is None else True}",
              f"Batch Size: {self.batch_size}", sep='\n')

class BERT_Model(nn.Module):
  
    def __init__(self, hparams):
        super(BERT_Model, self).__init__()

        self.name = hparams.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = 'bert-base-multilingual-cased'
        self.bert = BertModel.from_pretrained(model_name)

        lstm_input_dim = self.bert.config.hidden_size

        self.pos_embedding = nn.Embedding(
            hparams.pos_vocab_size, hparams.embedding_dim, padding_idx=0)
        self.pos_dropout = nn.Dropout(hparams.dropout)
            
        lstm_input_dim += hparams.embedding_dim    

        self.lstm = nn.LSTM(lstm_input_dim, hparams.hidden_dim, 
                            batch_first=True,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        self.dropout = nn.Dropout(hparams.dropout)

        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
  
    def forward(self, sequences, attention_mask, pos_sequences):
        
        bert_hidden_layer, _ = self.bert(
        input_ids=sequences,
        attention_mask=attention_mask
        )

        non_zeros = (sequences != 0).sum(dim=1).tolist()

        reconst = []
        for idx in range(len(non_zeros)):
            # print("no zeros", non_zeros[idx])
            # print("inputs ", sequences[idx])
            # print("sliced", sequences[idx, 1:non_zeros[idx]-1])
            reconst.append(bert_hidden_layer[idx, 1:non_zeros[idx]-1, :]) # correct one
        # print("reconst", reconst)
        # print("shape of every elem in 'reconst'", [t.shape for t in reconst])

        padded_again = torch.nn.utils.rnn.pad_sequence(reconst, batch_first=True, padding_value=0)
        # print("\n")
        # print("padded again shape", padded_again.shape)

        pos_embeddings = self.pos_embedding(pos_sequences)
        pos_embeddings_ = self.pos_dropout(pos_embeddings)
        # print("pos embed shape", pos_embeddings_.shape)
        
        embeds_ = torch.cat((padded_again, pos_embeddings_), dim=-1)
        # print("concat shape", embeds_.shape)


        o, _ = self.lstm(embeds_)
        o = self.dropout(o)
        logits = self.classifier(o)
        # print("logits / network output shape", logits.shape)
        # print("\n")

        return logits

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path):
        # print("\n\n bBAAAAAAAAAAAAAA ", path)
        # print("\n\n device is ", self.device)
        if self.device == 'cuda':
            # print("\n\n Loading from", path)
            # print("device is", self.device)
            model_state_dict = torch.load(path)
        else:
            model_state_dict = torch.load(path, map_location=self.device)
        model_name = 'bert-base-multilingual-cased'
        loaded_model = BertModel.from_pretrained(model_name, state_dict=model_state_dict)
        # self.load_state_dict(state_dict)
        return loaded_model

    def print_summary(self, show_weights=False, show_parameters=False):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        print(f"Params #: {'{:,}'.format(num_params)}")
        print('==================================================')



def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]

class Evaluator:
    # TODO: Check predict sentences torch.no_grad
    def __init__(self, model, test_dataset, t_data, idx2label):
        self.model = model
        self.model.eval()
        self.test_dataset = test_dataset
        self.micro_scores = None
        self.macro_scores = None
        self.class_scores = None
        self.confusion_matrix = None
        self.data = t_data
        self.idx2label = idx2label

    def compute_scores(self):
        original_len = list()
        all_predictions = list()
        all_labels = list()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # for step, samples in tqdm(enumerate(self.test_dataset), desc="Predicting batches of data", leave=False):
            for step, samples in (enumerate(self.test_dataset)):
                inputs, labels = samples['inputs'], samples['outputs']

                # print((labels != 0).sum(dim=1).tolist())
                original_len.extend((labels != 0).sum(dim=1).tolist())

                mask = (inputs != 0).to(device, dtype=torch.uint8)
                pos = samples['pos']
                if "BERT" or "bert" in self.model.name:
                    predictions = self.model(inputs, mask, pos)
                else:
                    predictions = self.model(inputs, pos)
                predictions = torch.argmax(predictions, -1).view(-1)
                labels = labels.view(-1)
                valid_indices = labels != 0
                valid_predictions = predictions[valid_indices]
                valid_labels = labels[valid_indices]
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())

        reconstructed_predictions = []                    
        var = 0
        for c_idx in range(len(original_len)):
            reconstructed_predictions.append(all_predictions[var:var+original_len[c_idx]])
            var += original_len[c_idx]

        eval_file = "{}/Prediction_vs_GroundTruth.txt".format(".")
        s_op_file = open(eval_file, "w", encoding="utf8")
        
        # for batch  in self.test_dataset:
            # for inp, label, prediction in zip(batch["inputs"], batch["outputs"], reconstructed_predictions):
        for sentence, label, prediction in zip(self.data.data_x, self.data.data_y, reconstructed_predictions): 
                
                # print(sentence, len(sentence))
                # print(label, len(label))
                # print(prediction, len(prediction))
                # print("\n")

                s_op_file.write("sen:")
                for token in sentence:
                    s_op_file.write(token + " ")
                s_op_file.write("\n")


                s_op_file.write("org:")
                for word in label:
                    s_op_file.write(word + " ")
                s_op_file.write("\n")

                s_op_file.write("pre:")
                for p in prediction:
                    s_op_file.write(str(self.idx2label[p]) + " ")
                s_op_file.write("\n")
                s_op_file.write("\n")
                
                assert len(label) == len(prediction)
        
        # global precision. Does take class imbalance into account.
        self.micro_scores = precision_recall_fscore_support(all_labels, all_predictions,
                                                            average="micro")

        # precision per class and arithmetic average of them. Does not take into account class imbalance.
        self.macro_scores = precision_recall_fscore_support(all_labels, all_predictions,
                                                            average="macro")

        self.class_scores = precision_score(all_labels, all_predictions,
                                            average=None)

        self.confusion_matrix = confusion_matrix(all_labels, all_predictions,
                                                 normalize='true')

    def pprint_confusion_matrix(self, conf_matrix):
        df_cm = pd.DataFrame(conf_matrix)
        fig = plt.figure(figsize=(10, 7))
        axes = fig.add_subplot(111)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=axes)  # font size
        axes.set_xlabel('Predicted labels')
        axes.set_ylabel('True labels')
        axes.set_title('Confusion Matrix')
        axes.xaxis.set_ticklabels(['B', 'I', 'O'])
        axes.yaxis.set_ticklabels(['B', 'I', 'O'])
        save_to = os.path.join(os.getcwd(), "model", f"{self.model.name}_confusion_matrix.png")
        plt.savefig(save_to)
        plt.show()

    def check_performance(self, idx2label):
        self.compute_scores()
        p, r, f, _ = self.macro_scores
        print("=" * 30)
        print(f'Macro Precision: {p:0.4f}, Macro Recall: {r:0.4f}, Macro F1 Score: {f:0.4f}')

        eval_file = "{}/Prediction_vs_GroundTruth.txt".format(".")
        s_op_file = open(eval_file, "a", encoding="utf8")
        s_op_file.write("=" * 30)
        s_op_file.write("\n")
        s_op_file.write(f'Macro Precision: {p:0.4f}, Macro Recall: {r:0.4f}, Macro F1 Score: {f:0.4f}')
        s_op_file.write("\n")

        print("=" * 30)
        print("Per class Precision:")
        # for idx_class, precision in sorted(enumerate(self.class_scores, start=0), key=lambda elem: -elem[1]):
        for idx_class, precision in enumerate(self.class_scores, start=1):
            # print("class score is", self.class_scores)
            # print(idx2label)
            # print("class is", idx_class)
            # if idx_class == 0:
            #     continue
            # else:
            label = idx2label[idx_class]
            print(f'{label}: {precision}')

        print("=" * 30)
        micro_p, micro_r, micro_f, _ = self.micro_scores
        print(f'Micro Precision: {micro_p:0.4f}, Micro Recall: {micro_r:0.4f}, Micro F1 Score: {micro_f:0.4f}')
        print("=" * 30)

        s_op_file.write(f'Micro Precision: {micro_p:0.4f}, Micro Recall: {micro_r:0.4f}, Micro F1 Score: {micro_f:0.4f}')
        s_op_file.write("\n")
        s_op_file.write("=" * 30)

        self.pprint_confusion_matrix(self.confusion_matrix)


def configure_workspace(SEED=1873337):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)



if __name__ == '__main__':

    pretrained_word_embeddings = True
    pretrained_pos_embeddings = True
    model_architecture = "bert" # baseline # crf

    configure_workspace()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_embeddings_ = None
    pos_embeddings_ = None

    # Set Hyper-parameters
    batch_size = 32 if model_architecture == "bert" else 128

    # name_ = 'BERT_Model_with_Bilingual_Embeddings_&_POS'
    # name_ = 'BERT_Grad_Clip_of_3_lr_of_5e7'
    name_ = 'BERT_Grad_Clip_of_3_lr_of_5e6'

    train_set_path = os.path.join(
        os.getcwd(), 'data', 'MWE_train_without_names.tsv')
    training_set = TSVDatasetParser(train_set_path, _device=device, tokenize_for_bert=True)
    word2idx_path = os.path.join(os.getcwd(), 'model', 'word_stoi.pkl')
    word2idx, idx2word = TSVDatasetParser.build_vocabulary(
        training_set.data_x, word2idx_path)

    pos2idx_path = os.path.join(os.getcwd(), 'model', 'pos_stoi.pkl')
    pos2idx, idx2pos = TSVDatasetParser.build_vocabulary(
        training_set.pos_x, pos2idx_path)
    label2idx, idx2label = TSVDatasetParser.encode_labels()
    training_set.encode_dataset(word2idx, label2idx, pos2idx)

    dev_set_path = os.path.join(
        os.getcwd(), 'data', 'MWE_dev_without_names.tsv')
    dev_set = TSVDatasetParser(dev_set_path, _device=device, tokenize_for_bert=True)
    dev_set.encode_dataset(word2idx, label2idx, pos2idx)

    hp = HyperParameters(name_, word2idx, label2idx, pos2idx,
                            pretrained_embeddings_,
                            pos_embeddings_, batch_size)
    # hp._print_info()

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=training_set, batch_size=batch_size,
                                collate_fn=TSVDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_set, batch_size=batch_size,
                                collate_fn=TSVDatasetParser.pad_batch)
    # Create and train model
    model = BERT_Model(hp).to(device)
    # model.print_summary()

    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

    trainer = BERT_Trainer(model=model, writer=writer_,
                            loss_function=CrossEntropyLoss(
                                ignore_index=label2idx['<PAD>']),
                            optimizer=Adam(model.parameters(), lr=5e-7),
                            label_vocab=label2idx)

    # save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model.pt")
    save_to_ = os.path.join(os.getcwd(), "model", f"{model.name}_ckpt_best.pth")
    print("\n\n\n\nSAVE TO\n\n\n", save_to_)
    # model = model._load(save_to_)
    # _ = trainer.train(train_dataset_, dev_dataset_, epochs=50, save_to=save_to_)
    
    # Load model in eval mode
    state_dict = torch.load(save_to_, map_location=device)
    model.load_state_dict(state_dict)    
    model.to(device)
    # # model.eval()

    evaluator = Evaluator(model, dev_dataset_, dev_set, idx2label)
    evaluator.check_performance(idx2label)