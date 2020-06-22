import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
try:
    from torchcrf import CRF
except ModuleNotFoundError:
    os.system("pip install pytorch-crf")
    from torchcrf import CRF


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.name = hparams.model_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.word_dropout = nn.Dropout(hparams.dropout)

        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.pos_embedding = nn.Embedding(
            hparams.pos_vocab_size, hparams.embedding_dim, padding_idx=0)
        self.pos_dropout = nn.Dropout(hparams.dropout)

        self.lstm = nn.LSTM(hparams.embedding_dim * 2, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, sequences, pos_sequence):
        embeddings = self.word_embedding(sequences)
        embeddings_ = self.word_dropout(embeddings)

        pos_embeddings = self.pos_embedding(pos_sequence)
        pos_embeddings_ = self.pos_dropout(pos_embeddings)

        embeds_ = torch.cat((embeddings_, pos_embeddings_), dim=2)

        o, _ = self.lstm(embeds_)
        o = self.dropout(o)
        logits = self.classifier(o)
        return logits

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def predict_sentences(self, tokens, words2idx, idx2label):
        raise NotImplementedError

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


class CRF_Model(nn.Module):
    def __init__(self, hparams):
        super(CRF_Model, self).__init__()
        self.name = hparams.model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.word_dropout = nn.Dropout(hparams.dropout)

        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
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
