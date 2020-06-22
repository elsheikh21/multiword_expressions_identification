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
