import json
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer


# @DatasetReader.register('StanceDetection')
class StanceDetectionReader(DatasetReader):
    """
    This is an abstract class to share the __init__-method and the
    text_to_instance-method between all readers of stance detection
    datasets.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_sequence_length = max_sequence_length

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as fin:
            for line in fin:
                instance_dict = json.loads(line)
                target = instance_dict['text1']
                text = instance_dict['text2']
                label = instance_dict['label_orig']
                yield self.text_to_instance(target, text, label)

    def text_to_instance(self, target: str, text: str, label: str = None) -> Instance:
        target_tokens = self.tokenizer.tokenize(target)
        text_tokens = self.tokenizer.tokenize(text)
        if len(target_tokens) + len(text_tokens) > self.max_sequence_length - 3:
            max_len_text_tokens = 509 - len(target_tokens)
            # TODO: remove hardcoded max-length (509) -> e.g. for bert-large models
            text_tokens = text_tokens[:max_len_text_tokens]
        combined_tokens = self.tokenizer.add_special_tokens(target_tokens, text_tokens)
        text_field = TextField(combined_tokens, self.token_indexers)
        fields = {'text_field': text_field}
        if label:
            fields['label'] = self.encode_label(label)
        return Instance(fields)

    def encode_label(self, stance: str) -> LabelField:
        raise NotImplementedError


@DatasetReader.register('SemEval2016')
class SemEval2016Reader(StanceDetectionReader):
    label_namespace = 'SemEval2016_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('IBMCS')
class IBMCSReader(StanceDetectionReader):
    label_namespace = 'IBMCS_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('arc')
class ArcReader(StanceDetectionReader):
    label_namespace = 'arc_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('ArgMin')
class ArgMinReader(StanceDetectionReader):
    label_namespace = 'ArgMin_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('FNC1')
class FNC1Reader(StanceDetectionReader):
    label_namespace = 'FNC1_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('IAC')
class IACReader(StanceDetectionReader):
    label_namespace = 'IAC_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('PERSPECTRUM')
class PERSPECTRUMReader(StanceDetectionReader):
    label_namespace = 'PERSPECTRUM_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SCD')
class SCDReader(StanceDetectionReader):
    label_namespace = 'SCD_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SemEval2019')
class SemEval2019Reader(StanceDetectionReader):
    label_namespace = 'SemEval2019_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('Snopes')
class SnopesReader(StanceDetectionReader):
    label_namespace = 'Snopes_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


DATASET_TO_READER = {
    'SemEval2016': SemEval2016Reader,
    'IBMCS': IBMCSReader,
    'arc': ArcReader,
    'ArgMin': ArgMinReader,
    'FNC1': FNC1Reader,
    'IAC': IACReader,
    'PERSPECTRUM': PERSPECTRUMReader,
    'SCD': SCDReader,
    'SemEval2019': SemEval2019Reader,
    'Snopes': SnopesReader
}
