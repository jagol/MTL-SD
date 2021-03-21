from typing import Dict, List, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models.heads import Head
from allennlp.modules.backbones import Backbone
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


@Backbone.register('sdmtl_backbone')
class SDMTLBackBone(Backbone):

    def __init__(self, encoder: BasicTextFieldEmbedder):
        super().__init__()
        self.encoder = encoder

    def forward(self, text_field: Dict[str, Dict[str, torch.Tensor]], label: torch.Tensor = None,
                encoder_name='bert') -> Dict[str, torch.Tensor]:
        return {
            'token_ids_encoded': self.encoder(text_field),
            'token_ids': text_field[encoder_name]['token_ids'],
            'mask': text_field[encoder_name]['mask'],
            'type_ids': text_field[encoder_name]['type_ids']
        }


@Head.register('stance_head')
class StanceHead(Head):

    default_predictor = 'head_predictor'

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, label_namespace: str,
                 dropout: float = 0.0, class_weights: Union[Dict[str, float], None] = None
                 ) -> None:
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, self.output_dim),
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }
        if class_weights:
            weights: List[float] = [0.0] * len(class_weights)
            for label, weight in class_weights.items():
                label_idx = self.vocab.get_token_index(label, namespace=label_namespace)
                weights[label_idx] = weight
            self.class_weights = torch.FloatTensor(class_weights)
            self.cross_ent = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.cross_ent = torch.nn.CrossEntropyLoss()

    def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        cls_tokens = token_ids_encoded[:, 0, :]
        # Shape: (batch_size, num_labels)
        logits = self.layers(cls_tokens)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        output = {'probs': probs}
        if label is not None:
            self.metrics['accuracy'](logits, label)
            self.metrics['f1_macro'](logits, label)
            output['loss'] = self.cross_ent(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }


@Head.register('stance_head_two_layers')
class StanceHeadTwoLayers(Head):

    default_predictor = 'head_predictor'

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, label_namespace: str,
                 dropout: float = 0.0, class_weights: Union[Dict[str, float], None] = None
                 ) -> None:
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.input_dim, output_dim)
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }
        if class_weights:
            weights: List[float] = [0.0] * len(class_weights)
            for label, weight in class_weights.items():
                label_idx = self.vocab.get_token_index(label, namespace=label_namespace)
                weights[label_idx] = weight
            self.class_weights = torch.FloatTensor(class_weights)
            self.cross_ent = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.cross_ent = torch.nn.CrossEntropyLoss()

    def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        cls_tokens = token_ids_encoded[:, 0, :]
        # Shape: (batch_size, num_labels)
        logits = self.layers(cls_tokens)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        output = {'probs': probs}
        if label is not None:
            self.metrics['accuracy'](logits, label)
            self.metrics['f1_macro'](logits, label)
            output['loss'] = self.cross_ent(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }


@Head.register('stance_head_sigmoid')
class StanceHeadSigmoid(Head):

    default_predictor = 'head_predictor'

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, dropout: float = 0.0,
                 class_weights: List[float] = None):
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, 1),
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }
        if class_weights:
            raise Exception('Class weights with BCEloss are not possible.')
            # self.class_weights = torch.FloatTensor(class_weights)
            # self.cross_ent = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.BCE_sigm_loss = torch.nn.BCEWithLogitsLoss()
        # self.label_regions = {}  # {upper_end: label}
        # self.range_per_label = 1 / self.output_dim
        # for i in range(self.output_dim):
        #     self.label_regions[self.range_per_label*(i+1)] = i

    # def float_to_label(self, float_num: torch.FloatTensor) -> torch.LongTensor:
    #     pass

    def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        cls_tokens = token_ids_encoded[:, 0, :]
        # Shape: (batch_size, num_labels)
        logits = torch.squeeze(self.layers(cls_tokens))

        output = {}
        if label is not None:
            pred_labels = []
            for num in logits:
                if num > 0.5:
                    pred_labels.append([1, 0])
                else:
                    pred_labels.append([0, 1])
            pred_labels = torch.LongTensor(pred_labels)
            self.metrics['accuracy'](pred_labels, label)
            self.metrics['f1_macro'](pred_labels, label)
            output['loss'] = self.BCE_sigm_loss(logits, label.float())
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }
