from typing import Dict, List

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

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, dropout: float = 0.0,
                 class_weights: List[float] = None):
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

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, dropout: float = 0.0,
                 class_weights: List[float] = None):
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
