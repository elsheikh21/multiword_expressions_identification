import importlib
import json
import logging
import os
import pickle
import random
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import nltk
import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from torch.nn import CrossEntropyLoss
from torch.nn.modules.module import _addindent
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from torchcrf import CRF
except ModuleNotFoundError:
    os.system("pip install pytorch-crf")
    from torchcrf import CRF


def parse_args():
    parser = ArgumentParser(description="MWEs Identification & Disambiguation")
    parser.add_argument('path')
    return vars(parser.parse_args())


def configure_workspace(SEED=1873337):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def save_pickle(save_to, save_what):
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def flat_list(l):
    return [_e for e in l for _e in e]


def fetch_sense_info(multiword_token, data=None, is_babelnet=True):
    lemmatizer = WordNetLemmatizer()
    lemma_word = lemmatizer.lemmatize(multiword_token.lower())
    if is_babelnet:
        if data is None:
            data = json.load(
                open(os.path.join(os.getcwd(), "it_en_lemma2synsets.json")))
        synsets = [val for key, val in data.items(
        ) if lemma_word.replace(' ', '_') in key]
        return ' '.join(flat_list(synsets))
    else:
        synsets = wn.synsets(lemma_word)
        if synsets is None or len(synsets) == 0:
            return " "

        sense_info = []
        for synset in synsets:
            sense_info.append(
                'wn:' + str(synset.offset()).zfill(8) + synset.pos())
        return ' '.join(sense_info)


def print_extracted_mwes(test_set, preds):
    for i in range(len(preds)):
        curr_sentence = test_set.data_x[i]
        prediction_list = preds[i][:len(curr_sentence)]
        sentence_id = curr_sentence[0]
        indices = [i for i, x in enumerate(prediction_list) if x == "B"]
        if indices == []:
            continue
        mwes = []
        for idx in indices:
            curr_mwes = []
            curr_mwes.append(idx)
            preds_list = prediction_list[idx:]
            for idx_ in range(len(preds_list)):
                if preds_list[idx_] == 'O':
                    curr_mwes.append(idx + idx_ - 1)
                    mwes.append(curr_mwes)
                    break
        for mwe in mwes:
            start_token = mwe[0]
            end_token = mwe[1] + 1
            mwe_ = ' '.join([curr_sentence[i]
                             for i in range(start_token, end_token)])
            sense_info = fetch_sense_info(mwe_)
            print(f'{sentence_id}\t{start_token}\t{end_token}\t{mwe_}\t{sense_info}')


class TSVDatasetParser(Dataset):
    def __init__(self, _path, _device, is_testing_data=False):
        self.encoded_data = []
        self.is_test_data = is_testing_data
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
                                               leave=False,
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
            for sentence, pos_sentence, labels in tqdm(zip(self.data_x, self.pos_x, self.data_y), desc='Indexing Data', leave=False, total=len(self.data_x)):
                data_x_stoi.append(torch.LongTensor(
                    [word2idx.get(word, 1) for word in sentence]).to(self.device))
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
        for pred in tqdm(predictions_, desc='Decoding Predictions', leave=False):
            predictions.append([idx2label.get(i) for i in pred])
        return predictions


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


class CRF_Model(nn.Module):
    def __init__(self, hparams):
        super(CRF_Model, self).__init__()
        self.name = hparams.model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.word_dropout = nn.Dropout(hparams.dropout)

        if hparams.embeddings is not None:
            # print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.pos_embedding = nn.Embedding(
            hparams.pos_vocab_size, hparams.embedding_dim, padding_idx=0)
        self.pos_dropout = nn.Dropout(hparams.dropout)

        if hparams.pos_embeddings is not None:
            print("initializing embeddings from pretrained")
            self.pos_embedding.weight.data.copy_(hparams.pos_embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim * 2, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        self.crf = CRF(hparams.num_classes, batch_first=True)

    def forward(self, sequences, pos_sequences):
        embeddings = self.word_embedding(sequences)
        embeddings_ = self.word_dropout(embeddings)

        pos_embeddings = self.pos_embedding(pos_sequences)
        pos_embeddings_ = self.pos_dropout(pos_embeddings)

        embeds_ = torch.cat((embeddings_, pos_embeddings_), dim=2)

        o, _ = self.lstm(embeds_)
        o = self.dropout(o)
        logits = self.classifier(o)
        return logits

    def log_probs(self, x, pos, tags, mask=None):
        emissions = self(x, pos)
        return self.crf(emissions, tags, mask=mask)

    def predict(self, x):
        emissions = self(x)
        return self.crf.decode(emissions)

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path):
        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def predict_sentences(self, test_dataset):
        self.eval()
        with torch.no_grad():
            all_predictions = []
            for step, samples in tqdm(enumerate(test_dataset), desc="Predicting",
                                      leave=False, total=len(test_dataset)):
                inputs, pos = samples['inputs'], samples['pos']
                predictions = self(inputs, pos)
                predictions = torch.argmax(predictions, -1).tolist()
                all_predictions.extend(predictions[:len(inputs)])
        return all_predictions

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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class WriterTensorboardX:
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        ensure_dir(writer_dir)
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module(
                    'tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'configs.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn(message)
                os.system('pip install tensorboardX')
                self.writer = importlib.import_module(
                    'tensorboardX').SummaryWriter(log_path)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image',
                                        'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

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
                    add_data('{}/{}'.format(self.mode, tag),
                             data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


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


class CRF_Trainer:
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
        for epoch in tqdm(range(1, epochs + 1), desc="Training", leave=False):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc='Fit On Batches',
                                     leave=False, total=len(train_dataset)):
                inputs, labels = sample['inputs'], sample['outputs']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                pos = sample['pos']
                self.optimizer.zero_grad()
                sample_loss = -self.model.log_probs(inputs, pos, labels, mask)
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(),
                                5.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            valid_loss = self.evaluate(valid_dataset)
            # print(f'| Epoch: {epoch:02} | Loss: {avg_epoch_loss:.4f} | Val Loss: {valid_loss:.4f} |')

            is_best = valid_loss <= best_val_loss
            if is_best:
                best_val_loss = valid_loss
                model_dir = os.path.join(
                    os.getcwd(), f'{self.model.name}_ckpt_best.pt')
                self.model._save(model_dir)

            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            scheduler.step(valid_loss)
            if es.step(valid_loss):
                # print(f"Early Stopping activated on epoch #: {epoch}")
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
                               leave=False, total=len(valid_dataset)):
                inputs = sample['inputs']
                labels = sample['outputs']
                pos = sample['pos']
                mask = (inputs != 0).to(self._device, dtype=torch.uint8)
                sample_loss = - \
                    self.model.log_probs(inputs, pos, labels, mask).sum()
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)


if __name__ == '__main__':
    # Get parser arguments (args)
    args = parse_args()
    path = args['path']

    # prepare to print to file
    sys.stdout.flush()

    # Download NLTK missing data
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    # Configure device for torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Configure seed and logging
    configure_workspace()

    # Read and index datasets
    train_set_path = os.path.join(os.getcwd(), 'MWE_train_without_names.tsv')
    training_set = TSVDatasetParser(train_set_path, _device=device)
    word2idx_path = os.path.join(os.getcwd(), 'word_stoi.pkl')
    word2idx, idx2word = TSVDatasetParser.build_vocabulary(training_set.data_x,
                                                           word2idx_path)

    pos2idx_path = os.path.join(os.getcwd(), 'pos_stoi.pkl')
    pos2idx, idx2pos = TSVDatasetParser.build_vocabulary(training_set.pos_x,
                                                         pos2idx_path)
    label2idx, idx2label = TSVDatasetParser.encode_labels()
    training_set.encode_dataset(word2idx, label2idx, pos2idx)

    dev_set_path = os.path.join(os.getcwd(), 'MWE_dev_without_names.tsv')
    dev_set = TSVDatasetParser(dev_set_path, _device=device)
    dev_set.encode_dataset(word2idx, label2idx, pos2idx)

    # Load pretrained word embeddings
    pretrained_path = os.path.join(os.getcwd(), 'vocab_embeddings_vector.npy')
    pretrained_embeddings_ = torch.from_numpy(np.load(pretrained_path))
    pos_embeddings_ = None

    # Set Hyper-parameters
    batch_size = 128
    name_ = 'CRF BiLSTM_II Model with Bilingual Embeddings & POS'
    hp = HyperParameters(name_, word2idx, label2idx, pos2idx,
                         pretrained_embeddings_,
                         pos_embeddings_, batch_size)

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=training_set, batch_size=batch_size,
                                collate_fn=TSVDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_set, batch_size=batch_size,
                              collate_fn=TSVDatasetParser.pad_batch)

    # Build model
    model = CRF_Model(hp).to(device)

    # Build training writer
    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

    # Build trainer
    trainer = CRF_Trainer(model=model, writer=writer_,
                          loss_function=CrossEntropyLoss(
                              ignore_index=label2idx['<PAD>']),
                          optimizer=Adam(model.parameters()),
                          label_vocab=label2idx)

    save_to_ = os.path.join(os.getcwd(), f"{model.name}_model.pt")
    _ = trainer.train(train_dataset_, dev_dataset_,
                      epochs=50, save_to=save_to_)

    test_set_path = os.path.join(os.getcwd(), path)
    test_set = TSVDatasetParser(test_set_path,
                                _device=device,
                                is_testing_data=True)
    test_set.encode_dataset(word2idx, label2idx, pos2idx)
    test_dataset_ = DataLoader(dataset=test_set, batch_size=batch_size,
                               collate_fn=TSVDatasetParser.pad_batch)
    predictions = model.predict_sentences(test_dataset_)
    preds = TSVDatasetParser.decode_predictions(predictions, idx2label)
    print_extracted_mwes(test_set, preds)
