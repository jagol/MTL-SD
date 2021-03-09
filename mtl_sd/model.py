from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.heads import Head
from allennlp.modules.backbones import Backbone
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
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

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, input_dim),
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }

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
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }


@Head.register('stance_head_two_layers')
class StanceHeadTwoLayers(Head):

    default_predictor = 'head_predictor'

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__(vocab=vocab)
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_dim, output_dim)
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }

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
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }


@Model.register('stance_classifier')
class StanceClassifier(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary = None):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.dr_ratio = 0.1
        num_labels = vocab.get_vocab_size('labels')
        self.head_one_layer = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.head_two_layers = torch.nn.Sequential(
            torch.nn.Dropout(self.dr_ratio),
            torch.nn.Linear(encoder.get_output_dim(), encoder.get_output_dim()),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(encoder.get_output_dim(), num_labels)
        )
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }

    def forward(self,
                text_field: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.embedder(text_field)
        # Shape: (batch_size, encoding_dim)
        cls_tokens = embedded_text[:, 0, :]
        # Shape: (batch_size, num_labels)
        logits = self.head_one_layer(cls_tokens)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.metrics['accuracy'](logits, label)
            self.metrics['f1_macro'](logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore'],
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset)
        }
