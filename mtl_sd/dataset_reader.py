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
                 label_type: str = 'label_orig',
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_sequence_length = max_sequence_length
        self.label_type = label_type

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as fin:
            for line in fin:
                instance_dict = json.loads(line)
                target = instance_dict['text1']
                text = instance_dict['text2']
                label = instance_dict[self.label_type]
                yield self.text_to_instance(target, text, label)

    def text_to_instance(self, target: str, text: str, label: str = None) -> Instance:
        target_tokens = self.tokenizer.tokenize(target)
        text_tokens = self.tokenizer.tokenize(text)
        if len(target_tokens) + len(text_tokens) > self.max_sequence_length - 3:
            max_len_text_tokens = (self.max_sequence_length - 3) - len(target_tokens)
            text_tokens = text_tokens[:max_len_text_tokens]
        combined_tokens = self.tokenizer.add_special_tokens(target_tokens, text_tokens)
        text_field = TextField(combined_tokens, self.token_indexers)
        fields = {'text_field': text_field}
        if label:
            fields['label'] = self.encode_label(label)
        return Instance(fields)

    def encode_label(self, stance: str) -> LabelField:
        raise NotImplementedError


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


@DatasetReader.register('IBMCS')
class IBMCSReader(StanceDetectionReader):
    label_namespace = 'IBMCS_labels'

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


@DatasetReader.register('SemEval2016Task6')
class SemEval2016Task6Reader(StanceDetectionReader):
    label_namespace = 'SemEval2016Task6_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SemEval2019Task7')
class SemEval2019Task7Reader(StanceDetectionReader):
    label_namespace = 'SemEval2019Task7_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('Snopes')
class SnopesReader(StanceDetectionReader):
    label_namespace = 'Snopes_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SemEval2016Task4B')
class SemEval2016Task4BReader(StanceDetectionReader):
    label_namespace = 'SemEval2016Task4B_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SemEval2016Task4C')
class SemEval2016Task4CReader(StanceDetectionReader):
    label_namespace = 'SemEval2016Task4C_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


DATASET_TO_READER = {
    'arc': ArcReader,
    'ArgMin': ArgMinReader,
    'FNC1': FNC1Reader,
    'IAC': IACReader,
    'IBMCS': IBMCSReader,
    'PERSPECTRUM': PERSPECTRUMReader,
    'SCD': SCDReader,
    'SemEval2016Task6': SemEval2016Task6Reader,
    'SemEval2019Task7': SemEval2019Task7Reader,
    'Snopes': SnopesReader,
    'SemEval2016Task4B': SemEval2016Task4BReader,
    'SemEval2016Task4C': SemEval2016Task4CReader,
}
