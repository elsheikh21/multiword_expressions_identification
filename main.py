import logging
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import (configure_workspace, load_bilingual_embeddings,
                   load_pos_embeddings)

from data_parser import TSVDatasetParser
from evaluator import Evaluator
from models import (BaselineModel, CRF_Model, BERT_Model, HyperParameters,
                    save_pos_embeddings, train_pos2vec)
from trainer import Trainer, CRF_Trainer, BERT_Trainer, WriterTensorboardX


if __name__ == '__main__':

    pretrained_word_embeddings = True
    pretrained_pos_embeddings = True
    model_architecture = "bert"  # baseline # crf

    configure_workspace()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_embeddings_ = None
    pos_embeddings_ = None

    if pretrained_word_embeddings:
        save_to = os.path.join(os.getcwd(), 'model',
                               'vocab_embeddings_vector.npy')
        embeddings_path = os.path.join(os.getcwd(), 'resources', 'wiki.en.vec')
        embeddings_path1 = os.path.join(
            os.getcwd(), 'resources', 'wiki.it.vec')
        pretrained_embeddings_ = load_bilingual_embeddings(embeddings_path,
                                                           embeddings_path1,
                                                           word2idx,
                                                           300,
                                                           save_to=save_to)

    if pretrained_pos_embeddings:
        # model = train_pos2vec(training_set.pos_x, 10, 300, 1e-3, 30)
        pos_embeddings_path = os.path.join(
            os.getcwd(), 'model', "pos_embeddings.npy")
        # save_pos_embeddings(model, pos_embeddings_path)
        pos_embeddings_ = load_pos_embeddings(
            pos_embeddings_path, pos2idx, 300)

    # Set Hyper-parameters
    batch_size = 4 if model_architecture == "bert" else 128
    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=training_set, batch_size=batch_size,
                                collate_fn=TSVDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_set, batch_size=batch_size,
                              collate_fn=TSVDatasetParser.pad_batch)

    if model_architecture == "baseline":
        name_ = 'BiLSTM Model'

        train_set_path = os.path.join(
            os.getcwd(), 'data', 'MWE_train_without_names.tsv')
        training_set = TSVDatasetParser(train_set_path, _device=device)
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
        dev_set = TSVDatasetParser(dev_set_path, _device=device)
        dev_set.encode_dataset(word2idx, label2idx, pos2idx)

        if pretrained_word_embeddings:
            save_to = os.path.join(os.getcwd(), 'model',
                                   'vocab_embeddings_vector.npy')
            embeddings_path = os.path.join(
                os.getcwd(), 'resources', 'wiki.en.vec')
            embeddings_path1 = os.path.join(
                os.getcwd(), 'resources', 'wiki.it.vec')
            pretrained_embeddings_ = load_bilingual_embeddings(embeddings_path,
                                                               embeddings_path1,
                                                               word2idx,
                                                               300,
                                                               save_to=save_to)

        if pretrained_pos_embeddings:
            # model = train_pos2vec(training_set.pos_x, 10, 300, 1e-3, 30)
            pos_embeddings_path = os.path.join(
                os.getcwd(), 'model', "pos_embeddings.npy")
            # save_pos_embeddings(model, pos_embeddings_path)
            pos_embeddings_ = load_pos_embeddings(
                pos_embeddings_path, pos2idx, 300)

        hp = HyperParameters(name_, word2idx, label2idx,
                             pos2idx, pretrained_embeddings_, batch_size)
        hp._print_info()

        # Create and train model
        model = BaselineModel(hp).to(device)
        model.print_summary()

        log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
        writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

        trainer = Trainer(model=model, writer=writer_,
                          loss_function=CrossEntropyLoss(
                              ignore_index=label2idx['<PAD>']),
                          optimizer=Adam(model.parameters()), epochs=50,
                          num_classes=hp.num_classes, verbose=True)

        save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model.pt")
        _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)

    elif model_architecture == "crf":
        name_ = 'CRF BiLSTM_II Model with Bilingual Embeddings & POS'

        train_set_path = os.path.join(
            os.getcwd(), 'data', 'MWE_train_without_names.tsv')
        training_set = TSVDatasetParser(train_set_path, _device=device)
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
        dev_set = TSVDatasetParser(dev_set_path, _device=device)
        dev_set.encode_dataset(word2idx, label2idx, pos2idx)

        if pretrained_word_embeddings:
            save_to = os.path.join(os.getcwd(), 'model',
                                   'vocab_embeddings_vector.npy')
            embeddings_path = os.path.join(
                os.getcwd(), 'resources', 'wiki.en.vec')
            embeddings_path1 = os.path.join(
                os.getcwd(), 'resources', 'wiki.it.vec')
            pretrained_embeddings_ = load_bilingual_embeddings(embeddings_path,
                                                               embeddings_path1,
                                                               word2idx,
                                                               300,
                                                               save_to=save_to)

        if pretrained_pos_embeddings:
            # model = train_pos2vec(training_set.pos_x, 10, 300, 1e-3, 30)
            pos_embeddings_path = os.path.join(
                os.getcwd(), 'model', "pos_embeddings.npy")
            # save_pos_embeddings(model, pos_embeddings_path)
            pos_embeddings_ = load_pos_embeddings(
                pos_embeddings_path, pos2idx, 300)

        hp = HyperParameters(name_, word2idx, label2idx, pos2idx,
                             pretrained_embeddings_,
                             pos_embeddings_, batch_size)
        hp._print_info()

        # Create and train model
        model = CRF_Model(hp).to(device)
        model.print_summary()

        log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
        writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

        trainer = CRF_Trainer(model=model, writer=writer_,
                              loss_function=CrossEntropyLoss(
                                  ignore_index=label2idx['<PAD>']),
                              optimizer=Adam(model.parameters()),
                              label_vocab=label2idx)

        save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model.pt")
        _ = trainer.train(train_dataset_, dev_dataset_,
                          epochs=50, save_to=save_to_)

    elif model_architecture == "bert":
        # name_ = 'BERT_Model_with_Bilingual_Embeddings_&_POS'
        # name_ = 'BERT_Grad_Clip_of_3_lr_of_5e7'
        name_ = 'BERT_Grad_Clip_of_3_lr_of_5e6'

        train_set_path = os.path.join(
            os.getcwd(), 'data', 'MWE_train_without_names.tsv')
        training_set = TSVDatasetParser(
            train_set_path, _device=device, tokenize_for_bert=True)
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
        dev_set = TSVDatasetParser(
            dev_set_path, _device=device, tokenize_for_bert=True)
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
        save_to_ = os.path.join(os.getcwd(), "model",
                                f"{model.name}_ckpt_best.pth")
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
