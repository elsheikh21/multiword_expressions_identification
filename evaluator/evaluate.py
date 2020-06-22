import os
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_fscore_support,
                             recall_score, f1_score)
from typing import List, Any


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


class Evaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.model.eval()
        self.test_dataset = test_dataset
        self.micro_scores = None
        self.macro_scores = None
        self.class_scores = None
        self.confusion_matrix = None

    def compute_scores(self):
        all_predictions = list()
        all_labels = list()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for step, samples in tqdm(enumerate(self.test_dataset), desc="Predicting batches of data", leave=False):
            inputs, labels = samples['inputs'], samples['outputs']
            pos = samples['pos']
            predictions = self.model(inputs, pos)
            predictions = torch.argmax(predictions, -1).view(-1)
            labels = labels.view(-1)
            valid_indices = labels != 0
            valid_predictions = predictions[valid_indices]
            valid_labels = labels[valid_indices]
            all_predictions.extend(valid_predictions.tolist())
            all_labels.extend(valid_labels.tolist())
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
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        save_to = os.path.join(os.getcwd(), "model", f"{self.model.name}_confusion_matrix.png")
        plt.savefig(save_to)
        plt.show()

    def check_performance(self, idx2label):
        self.compute_scores()
        p, r, f, _ = self.macro_scores
        print("=" * 30)
        print(f'Macro Precision: {p:0.4f}, Macro Recall: {r:0.4f}, Macro F1 Score: {f:0.4f}')

        print("=" * 30)
        print("Per class Precision:")
        for idx_class, precision in sorted(enumerate(self.class_scores, start=1), key=lambda elem: -elem[1]):
            label = idx2label[idx_class]
            print(f'{label}: {precision}')

        print("=" * 30)
        p, r, f, _ = self.micro_scores
        print(f'Micro Precision: {p:0.4f}, Micro Recall: {r:0.4f}, Micro F1 Score: {f:0.4f}')
        print("=" * 30)

        self.pprint_confusion_matrix(self.confusion_matrix)