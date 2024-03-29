from typing import Dict, List, Union, Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.models.heads import Head
from allennlp.modules.backbones import Backbone
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


@Backbone.register('sdmtl_backbone')
class SDMTLBackBone(Backbone):

    def __init__(self, encoder: BasicTextFieldEmbedder, encoder_name: str = 'bert'):
        super().__init__()
        self.encoder = encoder
        self.encoder_name = encoder_name

    def forward(self, text_field: Dict[str, Dict[str, torch.Tensor]], label: torch.Tensor = None,
                encoder_name='bert') -> Dict[str, torch.Tensor]:
        return {
            'token_ids_encoded': self.encoder(text_field),
            'token_ids': text_field[self.encoder_name]['token_ids'],
            'mask': text_field[self.encoder_name]['mask'],
            'type_ids': text_field[self.encoder_name]['type_ids']
        }


@Head.register('stance_head')
class StanceHead(Head):

    default_predictor = 'head_predictor'

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
            'f1_macro': self.metrics['f1_macro'].get_metric(reset=reset)['fscore']
        }


@Head.register('stance_head_crossent')
class StanceHeadCrossEnt(StanceHead):

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, label_namespace: str,
                 dropout: float = 0.0, class_weights: Union[Dict[str, float], None] = None,
                 use_sep_repr: bool = False) -> None:
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_sep_repr = use_sep_repr
        self.label_namespace = label_namespace + '_labels'
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }
        if class_weights:
            weights: List[float] = [0.0] * len(class_weights)
            for label, weight in class_weights.items():
                label_idx = self.vocab.get_token_index(label, namespace=self.label_namespace)
                weights[label_idx] = weight
            self.class_weights = torch.FloatTensor(weights)
            self.cross_ent = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.cross_ent = torch.nn.CrossEntropyLoss()

    def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        cls_repr = token_ids_encoded[:, 0, :]
        if self.use_sep_repr:
            last_sep_tokens = token_ids_encoded[:, -1, :]
            cls_repr = torch.cat([cls_repr, last_sep_tokens], dim=1)
        # Shape: (batch_size, num_labels)
        logits = self.layers(cls_repr)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        output = {'probs': probs}
        if label is not None:
            self.metrics['accuracy'](logits, label)
            self.metrics['f1_macro'](logits, label)
            output['loss'] = self.cross_ent(logits, label)
        return output


@Head.register('stance_head_crossent_1l')
class StanceHeadCrossEnt1L(StanceHeadCrossEnt):

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, label_namespace: str,
                 dropout: float = 0.0, class_weights: Union[Dict[str, float], None] = None,
                 use_sep_repr: bool = False) -> None:
        super().__init__(vocab=vocab, input_dim=input_dim, output_dim=output_dim,
                         label_namespace=label_namespace, dropout=dropout,
                         class_weights=class_weights, use_sep_repr=use_sep_repr)
        if use_sep_repr:
            self.input_dim = 2 * self.input_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, self.output_dim),
        )


@Head.register('stance_head_crossent_2l')
class StanceHeadCrossEnt2L(StanceHeadCrossEnt):

    def __init__(self, vocab: Vocabulary, input_dim: int, output_dim: int, label_namespace: str,
                 dropout: float = 0.0, class_weights: Union[Dict[str, float], None] = None,
                 use_sep_repr: bool = False) -> None:
        super().__init__(vocab=vocab, input_dim=input_dim, output_dim=output_dim,
                         label_namespace=label_namespace, dropout=dropout,
                         class_weights=class_weights, use_sep_repr=use_sep_repr)
        if use_sep_repr:
            self.input_dim = 2 * self.input_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.input_dim, output_dim)
        )


@Head.register('stance_head_mse')
class StanceHeadMSE(StanceHead):

    default_predictor = 'head_predictor'

    def __init__(self, vocab: Vocabulary, input_dim: int,
                 label_to_range: Dict[str, Tuple[float, float]], dropout: float = 0.0) -> None:
        super().__init__(vocab=vocab)
        self.input_dim = input_dim
        self.dropout = dropout
        self.label_to_range = label_to_range
        self.label_to_index = self._get_label_to_index()
        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'f1_macro': FBetaMeasure(average='macro')
        }
        self.mse_loss = torch.nn.MSELoss()

    def _get_label_to_index(self) -> Dict[str, int]:
        sorted_labels = sorted(self.label_to_range, key=lambda key: self.label_to_range[key])
        return {label: index for index, label in enumerate(sorted_labels)}

    def _get_pred_labels(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Args:
            logits: tensor of shape (batch-size, output-dims)
        """
        pred_labels = []
        logits_list = logits.tolist()
        if isinstance(logits_list, float):
            logits_list = [logits]
        for num in logits_list:
            pred_label = None
            for label in self.label_to_range:
                if self.label_to_range[label][0] <= num < self.label_to_range[label][1]:
                    if pred_label is None:
                        pred_label = label
            ohe_pred_label = len(self.label_to_range) * [0]
            label_index = self.label_to_index[pred_label]
            ohe_pred_label[label_index] = 1
            pred_labels.append(ohe_pred_label)
        return pred_labels

    def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        cls_tokens = token_ids_encoded[:, 0, :]
        # Shape: (batch_size, num_labels)
        logits = torch.squeeze(self.layers(cls_tokens))
        # get predicted labels
        pred_labels: List[List[int]] = self._get_pred_labels(logits)
        output = {'logits': logits, 'probs': pred_labels}
        if label is not None:
            pred_labels_tensor = torch.LongTensor(pred_labels)
            pred_labels_tensor_device = pred_labels_tensor.to(f'cuda:{logits.get_device()}')
            self.metrics['accuracy'](pred_labels_tensor_device, label)
            self.metrics['f1_macro'](pred_labels_tensor_device, label)
            output['loss'] = self.mse_loss(logits, label.float())
        return output


@Head.register('stance_head_mse_1l')
class StanceHeadMSE1L(StanceHeadMSE):

    def __init__(self, vocab: Vocabulary, input_dim: int,
                 label_to_range: Dict[str, Tuple[float, float]], dropout: float = 0.0) -> None:
        super().__init__(vocab=vocab, input_dim=input_dim, label_to_range=label_to_range,
                         dropout=dropout)
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, 1),
            torch.nn.Sigmoid()
        )


@Head.register('stance_head_mse_2l')
class StanceHeadMSE2L(StanceHeadMSE):

    def __init__(self, vocab: Vocabulary, input_dim: int,
                 label_to_range: Dict[str, Tuple[float, float]], dropout: float = 0.0) -> None:
        super().__init__(vocab=vocab, input_dim=input_dim, label_to_range=label_to_range,
                         dropout=dropout)
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.input_dim, 1),
            torch.nn.Sigmoid()
        )


# @Head.register('stance_head_ranking')
# class StanceHeadRanking(StanceHead):
#
#     def __init__(self, vocab: Vocabulary, input_dim: int,
#                  label_to_range: Dict[str, Tuple[float, float]], dropout: float = 0.0) -> None:
#         super().__init__(vocab=vocab)
#         self.input_dim = input_dim
#         self.dropout = dropout
#         self.label_to_range = label_to_range
#         self.label_to_index = self._get_label_to_index()
#         self.metrics = {
#             'accuracy': CategoricalAccuracy(),
#             'f1_macro': FBetaMeasure(average='macro')
#         }
#         self.mse_loss = torch.nn.MSELoss()
#
#     def _get_label_to_index(self) -> Dict[str, int]:
#         sorted_labels = sorted(self.label_to_range, key=lambda key: self.label_to_range[key])
#         return {label: index for index, label in enumerate(sorted_labels)}
#
#     def _get_pred_labels(self, logits: torch.Tensor) -> List[List[int]]:
#         """
#         Args:
#             logits: tensor of shape (batch-size, output-dims)
#         """
#         pred_labels = []
#         logits_list = logits.tolist()
#         if isinstance(logits_list, float):
#             logits_list = [logits]
#         for num in logits_list:
#             pred_label = None
#             for label in self.label_to_range:
#                 if self.label_to_range[label][0] <= num < self.label_to_range[label][1]:
#                     if pred_label is None:
#                         pred_label = label
#             ohe_pred_label = len(self.label_to_range) * [0]
#             label_index = self.label_to_index[pred_label]
#             ohe_pred_label[label_index] = 1
#             pred_labels.append(ohe_pred_label)
#         return pred_labels
#
#     def forward(self, token_ids_encoded: torch.Tensor, label: torch.Tensor = None
#                 ) -> Dict[str, torch.Tensor]:
#         # Shape: (batch_size, num_tokens, embedding_dim)
#         cls_tokens = token_ids_encoded[:, 0, :]
#         # Shape: (batch_size, num_labels)
#         logits = torch.squeeze(self.layers(cls_tokens))
#         # get predicted labels
#         pred_labels: List[List[int]] = self._get_pred_labels(logits)
#         output = {'logits': logits, 'probs': pred_labels}
#         if label is not None:
#             pred_labels_tensor = torch.LongTensor(pred_labels)
#             self.metrics['accuracy'](pred_labels_tensor, label)
#             self.metrics['f1_macro'](pred_labels_tensor, label)
#             output['loss'] = self.mse_loss(logits, label.float())
#         return output
