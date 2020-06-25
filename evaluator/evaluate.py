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