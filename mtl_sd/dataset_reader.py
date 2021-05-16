import json
import os
from typing import Dict, Iterable, List

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


@DatasetReader.register('UNIFIED')
class UnifiedReader(StanceDetectionReader):
    label_namespace = 'UNIFIED_labels'

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 label_type: str = 'label_orig',
                 corpora: List[str] = None,
                 train_file: str = None,
                 max_instances_per_corpus: int = 850,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_sequence_length = max_sequence_length
        self.label_type = label_type
        self.corpora = corpora
        self.train_file = train_file
        self.dev_file = 'dev.jsonl'
        self.max_instances_per_corpus = max_instances_per_corpus

    def _read(self, data_path: str) -> Iterable[Instance]:
        if data_path.endswith('train'):
            self.mode = 'train'
            data_path = data_path[:-5]
        elif data_path.endswith('dev'):
            self.mode = 'dev'
            data_path = data_path[:-3]
        else:
            raise Exception('Invalid datapath.')
        print(f'Mode: {self.mode}')
        if self.mode == 'train':
            files_open = [open(os.path.join(data_path, corpus, self.train_file)) for corpus in
                          self.corpora]
            instance_count = 0
            while instance_count < self.max_instances_per_corpus:
                for fopen in files_open:
                    line = fopen.readline()
                    if not line:
                        files_open.remove(fopen)
                    instance_dict = json.loads(line)
                    target = instance_dict['text1']
                    text = instance_dict['text2']
                    label = instance_dict[self.label_type]
                    yield self.text_to_instance(target, text, label)
                instance_count += 1
        else:
            if self.mode == 'dev':
                files_open = [open(os.path.join(data_path, corpus, self.dev_file)) for corpus in
                              self.corpora]
            elif self.mode == 'test':
                files_open = [open(os.path.join(data_path, corpus, self.test_file)) for corpus in
                              self.corpora]
            else:
                raise Exception('Wrong datapath name.')
            for fopen in files_open:
                for line in fopen:
                    instance_dict = json.loads(line)
                    target = instance_dict['text1']
                    text = instance_dict['text2']
                    label = instance_dict[self.label_type]
                    yield self.text_to_instance(target, text, label)

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


@DatasetReader.register('CoLA')
class CoLAReader(StanceDetectionReader):
    label_namespace = 'CoLA_labels'

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


@DatasetReader.register('IMDB')
class IMDBReader(StanceDetectionReader):
    label_namespace = 'IMDB_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('ISarcasm')
class ISarcasmReader(StanceDetectionReader):
    label_namespace = 'ISarcasm_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('MSRPara')
class MSRParaReader(StanceDetectionReader):
    label_namespace = 'MSRPara_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('MultiNLI')
class MultiNLIReader(StanceDetectionReader):
    label_namespace = 'MultiNLI_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('MultiTargetSD')
class MultiTargetSDReader(StanceDetectionReader):
    label_namespace = 'MultiTargetSD_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('PERSPECTRUM')
class PERSPECTRUMReader(StanceDetectionReader):
    label_namespace = 'PERSPECTRUM_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('QQP')
class QQPReader(StanceDetectionReader):
    label_namespace = 'QQP_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('RTE')
class RTEReader(StanceDetectionReader):
    label_namespace = 'RTE_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SCD')
class SCDReader(StanceDetectionReader):
    label_namespace = 'SCD_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('SemEval2016Task4A')
class SemEval2016Task4AReader(StanceDetectionReader):
    label_namespace = 'SemEval2016Task4A_labels'

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


@DatasetReader.register('SST')
class SSTReader(StanceDetectionReader):
    label_namespace = 'SST_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('STSB')
class STSBReader(StanceDetectionReader):
    label_namespace = 'STSB_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('TargetDepSA')
class TargetDepSAReader(StanceDetectionReader):
    label_namespace = 'TargetDepSA_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


@DatasetReader.register('WNLI')
class WNLIReader(StanceDetectionReader):
    label_namespace = 'WNLI_labels'

    def encode_label(self, stance: str) -> LabelField:
        return LabelField(stance, self.label_namespace)


DATASET_TO_READER = {
    'arc': ArcReader,
    'ArgMin': ArgMinReader,
    'CoLA': CoLAReader,
    'FNC1': FNC1Reader,
    'IAC': IACReader,
    'IBMCS': IBMCSReader,
    'IMDB': IMDBReader,
    'ISarcasm': ISarcasmReader,
    'MSRPara': MSRParaReader,
    'MultiNLI': MultiNLIReader,
    'MultiTargetSD': MultiTargetSDReader,
    'PERSPECTRUM': PERSPECTRUMReader,
    'QQP': QQPReader,
    'RTE': RTEReader,
    'SCD': SCDReader,
    'SemEval2016Task4A': SemEval2016Task4AReader,
    'SemEval2016Task4B': SemEval2016Task4BReader,
    'SemEval2016Task4C': SemEval2016Task4CReader,
    'SemEval2016Task6': SemEval2016Task6Reader,
    'SemEval2019Task7': SemEval2019Task7Reader,
    'Snopes': SnopesReader,
    'SST': SSTReader,
    'STSB': STSBReader,
    'UNIFIED': UnifiedReader,
    'TargetDepSA': TargetDepSAReader,
    'WNLI': WNLIReader,
}
