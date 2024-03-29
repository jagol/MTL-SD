import os
import re
from os.path import join as osjoin
import csv
import json
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, DefaultDict, Set, Union

import numpy as np


"""
Preprocess a corpus.

Given a corpus do:
- create devset and/or testset, if they do not yet exist,
- map labels to unified labelset, where possible, 
    (two labelssets, a original and a unified will be in the output)
- convert to jsonl format.

The devset is taken from the trainset if it does not yet exist.

Usage:
python3 preprocess.py -c <corpus-name> -s <dev-size-relative-to-train> -d <path_data_dir>
"""


instances_type = List[Dict[str, str]]
tws_type = Dict[str, Tuple[str, str]]


NO_TARGET_TOKEN = "None"


class LabelsUnified:
    PRO = 'pro'
    CON = 'con'
    OTHER = 'other'
    NEUTRAL = 'neutral'
    UNRELATED = 'unrelated'
    DISCUSS = 'discuss'
    NONE = 'none'
    NOARGUMENT = 'NoArgument'
    QUERY = 'query'


class Fields:
    ID = 'id'
    TEXT1 = 'text1'  # usually target
    TEXT2 = 'text2'  # usually comment
    LABEL_ORIGINAL = 'label_orig'
    LABEL_UNIFIED = 'label_uni'
    TASK = 'task'


class PreProcessor:
    file_names_out = {
        'train': 'train.jsonl',
        'dev': 'dev.jsonl',
        'test': 'test.jsonl'
    }
    corpus_dir = ''  # to be overwritten by dataset specific processors

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.corpus_path = osjoin(self.data_dir, self.corpus_dir)
        self.instance_id = 0

    def process(self, dev_size: float) -> None:
        """Main method of all corpus processors.

        Needs to be implemented by the individual corpus processors.

        Args:
            dev_size: 0-1, proportion of train-instances used for
                devset. Ignored if devset already exists.
        """
        raise NotImplementedError

    @staticmethod
    def _write_to_jsonlfile(instances: instances_type, fpath: str) -> None:
        with open(fpath, 'w') as fout:
            for instance in instances:
                fout.write(json.dumps(instance) + '\n')

    @staticmethod
    def _split_train_dev_set(train_dev: List[Any], dev_size: float) -> Tuple[List[Any], List[Any]]:
        """Split train-dev-set into train- and dev-set.

        Args:
            train_dev: list of instances that is split
            dev_size: between 0 and 1, proportion of instances allocated
                to dev-set

        Returns:
            {'train': train-instances, 'dev': dev-instances}
        """
        random.shuffle(train_dev)
        split_index = int(len(train_dev) * (1-dev_size))
        return train_dev[:split_index], train_dev[split_index:]


class SemEval2016Task6Processor(PreProcessor):
    file_names_in = {
        'train': 'trainingdata-all-annotations.txt',
        'dev': 'trialdata-all-annotations.txt',
        'test': 'testdata-taskA-all-annotations.txt'
    }
    corpus_dir = 'SemEval2016Task6/'
    label_mapping = {
        'FAVOR': LabelsUnified.PRO,
        'AGAINST': LabelsUnified.CON,
        'NONE': LabelsUnified.NONE
    }

    def process(self, dev_size: float) -> None:
        self._process_file(self.file_names_in['train'], self.file_names_out['train'])
        self._process_file(self.file_names_in['dev'], self.file_names_out['dev'])
        self._process_file(self.file_names_in['test'], self.file_names_out['test'])

    def _process_file(self, fname_in, fname_out):
        fpath_in = osjoin(self.corpus_path, fname_in)
        fpath_out = osjoin(self.corpus_path, fname_out)
        with open(fpath_in, encoding='latin1') as fin, open(fpath_out, 'w') as fout:
            reader = csv.DictReader(fin, delimiter='\t')
            for row in reader:
                fout.write(json.dumps({
                    Fields.ID: row['ID'],
                    Fields.TEXT1: row['Target'],
                    Fields.TEXT2: row['Tweet'],
                    Fields.LABEL_ORIGINAL: row['Stance'],
                    Fields.LABEL_UNIFIED: self.label_mapping[row['Stance']],
                    Fields.TASK: 'SemEval2016Task6'
                }) + '\n')


class IBMCSProcessor(PreProcessor):
    file_names_in = {
        'train_test': 'claim_stance_dataset_v1.csv',
    }
    corpus_dir = 'IBMCS/'
    label_mapping = {
        'PRO': LabelsUnified.PRO,
        'CON': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        fin = open(osjoin(self.corpus_path, self.file_names_in['train_test']))
        reader = csv.DictReader(fin)
        next(reader)  # skip header
        train_instances = []
        test_instances = []
        id_ = 0
        for row in reader:
            if row['split'] == 'train':
                train_instances.append({
                    Fields.ID: id_,
                    Fields.TEXT1: row['topicTarget'],
                    Fields.TEXT2: row['claims.claimCorrectedText'],
                    Fields.LABEL_ORIGINAL: row['claims.stance'],
                    Fields.LABEL_UNIFIED: self.label_mapping[row['claims.stance']],
                    Fields.TASK: 'IBMCS'
                })
                id_ += 1
            elif row['split'] == 'test':
                test_instances.append({
                    Fields.ID: id_,
                    Fields.TEXT1: row['topicTarget'],
                    Fields.TEXT2: row['claims.claimCorrectedText'],
                    Fields.LABEL_ORIGINAL: row['claims.stance'],
                    Fields.LABEL_UNIFIED: self.label_mapping[row['claims.stance']],
                    Fields.TASK: 'IBMCS'
                })
                id_ += 1
            else:
                raise Exception(f"Row contains unknown value for 'split': {row['split']}")
        fin.close()
        train_instances, dev_instances = self._split_train_dev_set(train_instances, dev_size)
        self._write_to_jsonlfile(train_instances, osjoin(self.corpus_path,
                                                         self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances, osjoin(self.corpus_path,
                                                       self.file_names_out['dev']))
        self._write_to_jsonlfile(test_instances, osjoin(self.corpus_path,
                                                        self.file_names_out['test']))


class ArcProcessor(PreProcessor):
    file_names_in = {
        'bodies': 'arc_bodies.csv',
        'train': 'arc_stances_train.csv',
        'test': 'arc_stances_test.csv'
    }
    corpus_dir = 'arc/'
    label_mapping = {
        'agree': LabelsUnified.PRO,
        'disagree': LabelsUnified.CON,
        'unrelated': LabelsUnified.UNRELATED,
        'discuss': LabelsUnified.DISCUSS
    }

    def __init__(self, data_dir: str):
        super().__init__(data_dir)

    def process(self, dev_size: float) -> None:
        bodies: Dict[str, str] = {}
        with open(osjoin(self.corpus_path, self.file_names_in['bodies'])) as fin:
            body_reader = csv.reader(fin)
            for body_id, body in body_reader:
                bodies[body_id] = body
        train_dev_set = self._load_arc(osjoin(self.corpus_path, self.file_names_in['train']))
        testset = self._load_arc(osjoin(self.corpus_path, self.file_names_in['test']))
        trainset, devset = self._split_train_dev_set(train_dev_set, dev_size)
        train_instances = self._merge_bodies_with_split(bodies, trainset)
        dev_instances = self._merge_bodies_with_split(bodies, devset)
        test_instances = self._merge_bodies_with_split(bodies, testset)
        self._write_to_jsonlfile(train_instances, osjoin(self.corpus_path,
                                                         self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances, osjoin(self.corpus_path,
                                                       self.file_names_out['dev']))

        self._write_to_jsonlfile(test_instances, osjoin(self.corpus_path,
                                                        self.file_names_out['test']))

    @staticmethod
    def _load_arc(fpath: str) -> List[Tuple[str, str, str]]:
        arcset = []
        with open(fpath) as fin:
            reader = csv.reader(fin)
            next(reader)  # skip header
            for headline, body_id, stance in reader:
                arcset.append((body_id, headline, stance))
        return arcset

    def _merge_bodies_with_split(self,
                                 bodies: Dict[str, str],
                                 arcset: List[Tuple[str, str, str]]
                                 ) -> instances_type:
        bodies_with_instances = []
        for body_id, headline, stance in arcset:
            bodies_with_instances.append({
                Fields.ID: body_id,
                Fields.TEXT1: headline,
                Fields.TEXT2: bodies[body_id],
                Fields.LABEL_ORIGINAL: stance,
                Fields.LABEL_UNIFIED: self.label_mapping[stance],
                Fields.TASK: 'arc'
            })
        return bodies_with_instances


class ArgMinProcessor(PreProcessor):
    corpus_dir = 'ArgMin/'
    label_mapping = {
        'Argument_for': LabelsUnified.PRO,
        'Argument_against': LabelsUnified.CON,
        'NoArgument': LabelsUnified.NOARGUMENT,  # Should never be used, since filtered out.
    }

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.train_set = []
        self.dev_set = []
        self.test_set = []

    def process(self, dev_size: float) -> None:
        data_path = osjoin(self.corpus_path, 'data/')
        for fname in os.listdir(data_path):
            self.load_from_file(osjoin(data_path, fname))
        self._write_to_jsonlfile(self.train_set,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(self.dev_set,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(self.test_set,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    def load_from_file(self, fpath: str):
        with open(fpath) as f:
            next(f)
            for line in f:
                fields = line.strip('\n').split('\t')
                if fields[5] == 'NoArgument':
                    continue
                instance = {
                    Fields.ID: self.instance_id,
                    Fields.TEXT1: fields[0],  # row['topic'],
                    Fields.TEXT2: fields[4],  # row['sentence'],
                    Fields.LABEL_ORIGINAL: fields[5],  # row['annotation'],
                    Fields.LABEL_UNIFIED: self.label_mapping[fields[5]],  # row['annotation']]
                    Fields.TASK: 'ArgMin'
                }
                self.instance_id += 1
                if fields[6] == 'train':
                    self.train_set.append(instance)
                elif fields[6] == 'val':
                    self.dev_set.append(instance)
                elif fields[6] == 'test':
                    self.test_set.append(instance)
                else:
                    raise Exception(f'Error. Set {fields[6]} unknown.')


class CoLAProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/CoLA/train.tsv',
        'dev': 'data/CoLA/dev.tsv',
        'test': 'data/CoLA/test.tsv',
    }
    file_names_out = {
        'train': 'data/CoLA/train.jsonl',
        'dev': 'data/CoLA/dev.jsonl',
        'test': 'data/CoLA/test.jsonl',
    }
    corpus_dir = 'CoLA/'
    label_mapping = {
        '1': LabelsUnified.PRO,
        '0': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        train_test_set = self._load(self.file_names_in['train'])
        train_set, test_set = self._split_train_dev_set(train_test_set, dev_size)
        dev_set = self._load(self.file_names_in['dev'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            next(fin)
            for line in fin:
                source, label, anno, sent = line.strip('\n').split('\t')
                instances.append({
                    Fields.ID: self.instance_id,
                    Fields.TEXT1: NO_TARGET_TOKEN,
                    Fields.TEXT2: sent,
                    Fields.LABEL_ORIGINAL: label,
                    Fields.LABEL_UNIFIED: self.label_mapping[label],
                    Fields.TASK: 'CoLA'
                })
                self.instance_id += 1
        return instances


class FNC1Processor(PreProcessor):
    corpus_dir = 'FNC1/'
    label_mapping = {
        'unrelated': LabelsUnified.UNRELATED,
        'discuss': LabelsUnified.DISCUSS,
        'agree': LabelsUnified.PRO,
        'disagree': LabelsUnified.CON
    }

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.train_bodies = []
        self.test_bodies = []
        self.train_stances = []
        self.train_stances = []
        self.dev_stances = []

    def process(self, dev_size: float) -> None:
        train_bodies = self.load_bodies(osjoin(self.corpus_path, 'train_bodies.csv'))
        test_bodies = self.load_bodies(osjoin(self.corpus_path,
                                              'competition_test_bodies.csv'))
        train_dev_instances = self.load_instances(
            osjoin(self.corpus_path, 'train_stances.csv'), train_bodies)
        test_instances = self.load_instances(
            osjoin(self.corpus_path, 'competition_test_stances.csv'), test_bodies)
        train_instances, dev_instances = self._split_train_dev_set(train_dev_instances, dev_size)
        self._write_to_jsonlfile(train_instances,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(test_instances,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    @staticmethod
    def load_bodies(fpath: str) -> Dict[str, str]:
        with open(fpath) as fin:
            reader = csv.reader(fin)
            next(reader)
            return {body_id: article_body for body_id, article_body in reader}

    def load_instances(self, fpath: str, bodies: Dict[str, str]) -> instances_type:
        instances = []
        with open(fpath) as fin:
            reader = csv.reader(fin)
            next(reader)
            for row in reader:
                instances.append({
                    Fields.ID: self.instance_id,
                    Fields.TEXT1: row[0],
                    Fields.TEXT2: bodies[row[1]],
                    Fields.LABEL_ORIGINAL: row[2],
                    Fields.LABEL_UNIFIED: self.label_mapping[row[2]],
                    Fields.TASK: 'FNC1'
                })
                self.instance_id += 1
        return instances


class IACProcessor(PreProcessor):
    """Code for this class partly taken over/copied/adjusted from:
    https://github.com/UKPLab/mdl-stance-robustness/blob/master/data_utils/glue_utils.py#L463

    Important: Splits (`topic_set_dict`) are reused for comparability.
    """

    corpus_dir = 'IAC/'
    label_mapping = {
        'pro': LabelsUnified.PRO,
        'anti': LabelsUnified.CON,
        'other': LabelsUnified.OTHER,
    }

    topic_set_dict = {
        'evolution': 'train',
        'death penalty': 'train',
        'gay marriage': 'train',
        'climate change': 'dev',
        'gun control': 'train',
        'healthcare': 'train',
        'abortion': 'train',
        'existence of god': 'test',
        'communism vs capitalism': 'dev',
        'marijuana legalization': 'test'
    }
    stance_mapping = {0: 'pro', 1: 'anti', 2: 'other'}

    def __init__(self, data_dir: str):
        super().__init__(data_dir)

    def process(self, dev_size: float) -> None:
        author_data = self._load_author_data(osjoin(self.corpus_path, 'author_stance.csv'))
        discussions = self._load_discussions([d['discussion_id'] for d in author_data])
        instances = self._merge(author_data, discussions)
        train_instances, dev_instances, test_instances = self._split_train_dev_test_set(instances)
        self._write_to_jsonlfile(train_instances,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(test_instances,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    def _merge(self,
               stances: List[Dict[str, str]],
               discussions: Dict[str, DefaultDict[str, List[str]]]) -> instances_type:
        instances = []
        for author_data in stances:
            discussion_id = author_data['discussion_id']
            text = ' '.join([s for s in discussions[discussion_id][author_data['author']]])
            instance = {
                Fields.ID: self.instance_id,
                Fields.TEXT1: author_data['topic'],
                Fields.TEXT2: text,
                Fields.LABEL_ORIGINAL: author_data['stance'],
                Fields.LABEL_UNIFIED: self.label_mapping[author_data['stance']],
                Fields.TASK: 'IAC'
            }
            instances.append(instance)
            self.instance_id += 1
        return instances

    def _load_author_data(self, fpath: str):
        author_data = []
        with open(fpath) as fin:
            reader = csv.DictReader(fin)
            for row in reader:  # topic,discussion_id,author,pro,anti,other
                overall_stance_idx = int(np.argmax([int(row['pro']), int(row['anti']),
                                                    int(row['other'])]))
                author_data.append({
                    'topic': row['topic'],
                    'discussion_id': row['discussion_id'],
                    'author': row['author'],
                    'stance': self.stance_mapping[overall_stance_idx]
                })
        return author_data

    def _load_discussions(self, discussion_ids: List[str]
                          ) -> Dict[str, DefaultDict[str, List[str]]]:
        """Roughly written after:
        https://github.com/UKPLab/mdl-stance-robustness/blob/master/data_utils/glue_utils.py#L498
        """
        discussions = {}
        for id_ in discussion_ids:
            with open(osjoin(self.corpus_path, f'discussions/{id_}.json')) as f:
                discussion = json.load(f)
                user_posts = defaultdict(list)
                for post in discussion[0]:
                    user_posts[post[2]].append(post[3])
                discussions[id_] = user_posts
        return discussions

    def _split_train_dev_test_set(self, instances: instances_type
                                  ) -> Tuple[instances_type, instances_type, instances_type]:
        train = []
        dev = []
        test = []
        for instance in instances:
            if self.topic_set_dict[instance['text1']] == 'train':  # text1 is topic
                train.append(instance)
            elif self.topic_set_dict[instance['text1']] == 'dev':
                dev.append(instance)
            elif self.topic_set_dict[instance['text1']] == 'test':
                test.append(instance)
        return train, dev, test


class IMDBProcessor(PreProcessor):
    corpus_dir = 'IMDB'
    label_mapping = {
        'pos': LabelsUnified.PRO,
        'neg': LabelsUnified.CON,
    }
    file_names_out = {
        'train': 'data/IMDB/train.jsonl',
        'dev': 'data/IMDB/dev.jsonl',
        'test': 'data/IMDB/test.jsonl',
    }

    def process(self, dev_size: float) -> None:
        train_pos = self._load(osjoin(self.corpus_path, 'train/pos/'))
        train_neg = self._load(osjoin(self.corpus_path, 'train/neg/'))
        test_pos = self._load(osjoin(self.corpus_path, 'test/pos/'))
        test_neg = self._load(osjoin(self.corpus_path, 'test/neg/'))
        train_set, dev_set = self._split_train_dev_set(train_pos + train_neg, dev_size)
        test_set = test_pos + test_neg
        random.shuffle(test_set)
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fdir: str) -> instances_type:
        instances = []
        label = 'pos' if 'pos' in fdir else 'neg'
        for fname in os.listdir(fdir):
            with open(osjoin(fdir, fname)) as fin:
                instances.append({
                    Fields.ID: self.instance_id,
                    Fields.TEXT1: NO_TARGET_TOKEN,
                    Fields.TEXT2: fin.read().strip(),
                    Fields.LABEL_ORIGINAL: label,
                    Fields.LABEL_UNIFIED: self.label_mapping[label],
                    Fields.TASK: 'IMDB'
                })
                self.instance_id += 1
        return instances


class ISarcasmProcessor(PreProcessor):
    corpus_dir = 'ISarcasm'
    label_mapping = {
        'sarcastic': LabelsUnified.CON,
        'not_sarcastic': LabelsUnified.PRO,
    }
    file_names_in = {
        'train_labels': 'data/ISarcasm/isarcasm_train.csv',
        'train_tweets': 'data/ISarcasm/train_tweets.csv',
        'test_labels': 'data/ISarcasm/isarcasm_test.csv',
        'test_tweets': 'data/ISarcasm/test_tweets.csv',
    }
    file_names_out = {
        'train': 'data/ISarcasm/train.jsonl',
        'dev': 'data/ISarcasm/dev.jsonl',
        'test': 'data/ISarcasm/test.jsonl',
    }

    def process(self, dev_size: float) -> None:
        train_dev_set = self._load(self.file_names_in['train_labels'],
                                   self.file_names_in['train_tweets'])
        test_set = self._load(self.file_names_in['test_labels'],
                              self.file_names_in['test_tweets'])
        train_set, dev_set = self._split_train_dev_set(train_dev_set, dev_size)
        random.shuffle(test_set)
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_labels: str, fpath_tweets: str) -> instances_type:
        instances = []
        with open(fpath_labels) as flabels, open(fpath_tweets) as ftweets:
            label_reader = csv.reader(flabels)
            next(label_reader)
            tweet_reader = csv.reader(ftweets)
            id_to_label = {id_: label for id_, label, sublabel in label_reader}
            for id_, tweet in tweet_reader:
                label_orig = id_to_label[id_]
                instances.append({
                    Fields.ID: id_,
                    Fields.TEXT1: NO_TARGET_TOKEN,
                    Fields.TEXT2: tweet,
                    Fields.LABEL_ORIGINAL: label_orig,
                    Fields.LABEL_UNIFIED: self.label_mapping[label_orig],
                    Fields.TASK: 'ISarcasm'
                })
        return instances


class MSRParaProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/MSRPara/msr_paraphrase_train.txt',
        'test': 'data/MSRPara/msr_paraphrase_test.txt',
    }
    file_names_out = {
        'train': 'data/MSRPara/train.jsonl',
        'dev': 'data/MSRPara/dev.jsonl',
        'test': 'data/MSRPara/test.jsonl',
    }
    corpus_dir = 'MSRPara/'
    label_mapping = {
        '1': LabelsUnified.PRO,
        '0': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        train_dev_set = self._load(self.file_names_in['train'])
        train_set, dev_set = self._split_train_dev_set(train_dev_set, dev_size)
        test_set = self._load(self.file_names_in['test'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            next(fin)
            for line in fin:
                label, idx1, idx2, sent1, sent2 = line.strip('\n').split('\t')
                instances.append({
                    Fields.ID: f'{idx1}_{idx2}',
                    Fields.TEXT1: sent1,
                    Fields.TEXT2: sent2,
                    Fields.LABEL_ORIGINAL: label,
                    Fields.LABEL_UNIFIED: self.label_mapping[label],
                    Fields.TASK: 'MSRPara'
                })
        return instances


class MultiNLIProcessor(PreProcessor):
    corpus_dir = 'MultiNLI'
    label_mapping = {
        'neutral': LabelsUnified.OTHER,
        'entailment': LabelsUnified.PRO,
        'contradiction': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        ptrain_in = osjoin(self.corpus_path, 'multinli_1.0_train.jsonl')
        ptrain_out = osjoin(self.corpus_path, 'train.jsonl')
        pdev_in_matched = osjoin(self.corpus_path, 'multinli_1.0_dev_matched.jsonl')
        pdev_in_mismatched = osjoin(self.corpus_path, 'multinli_1.0_dev_mismatched.jsonl')
        pdev_out = osjoin(self.corpus_path, 'dev.jsonl')
        self._process([ptrain_in], ptrain_out)
        self._process([pdev_in_matched, pdev_in_mismatched], pdev_out)

    def _process(self, paths_in: List[str], path_out: str):
        fout = open(path_out, 'w')
        for path in paths_in:
            with open(path) as fin:
                for line in tqdm(fin):
                    instance = json.loads(line)
                    if instance['gold_label'] not in self.label_mapping.keys():
                        continue
                    fout.write(json.dumps({
                        Fields.ID: instance['pairID'],
                        Fields.TEXT1: instance['sentence1'],
                        Fields.TEXT2: instance['sentence2'],
                        Fields.LABEL_ORIGINAL: instance['gold_label'],
                        Fields.LABEL_UNIFIED: self.label_mapping[instance['gold_label']],
                        Fields.TASK: 'MultiNLI'
                    }) + '\n')


class MultiTargetSDProcessor(PreProcessor):
    corpus_dir = 'MultiTargetSD'
    label_mapping = {
        'FAVOR': LabelsUnified.PRO,
        'AGAINST': LabelsUnified.CON,
        'NONE': LabelsUnified.NONE
    }
    file_names_in = {
        'tweets': 'multi_target_sd_tweets.csv',
        'labels': 'all_data_tweet_id.txt'
    }
    file_names_out = {
        'train': 'data/MultiTargetSD/train.jsonl',
        'dev': 'data/MultiTargetSD/dev.jsonl',
        'test': 'data/MultiTargetSD/test.jsonl',
    }

    def process(self, dev_size: float) -> None:
        train_set, dev_set, test_set = self._load_tweets(
            tweets_file=osjoin(self.corpus_path, self.file_names_in['tweets']),
            label_file=osjoin(self.corpus_path, self.file_names_in['labels'])
        )
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load_tweets(self, tweets_file: str, label_file: str
                     ) -> Tuple[instances_type, instances_type, instances_type]:
        train_set, dev_set, test_set = [], [], []
        with open(tweets_file) as tf, open(label_file) as lf:
            tweet_reader = csv.reader(tf)
            label_reader = csv.reader(lf)
            tweet_dict = {}
            label_dict = {}
            next(label_reader)  # skip header
            for row in label_reader:
                label_dict[row[0]] = {
                    'target1': row[1],
                    'stance1': row[2],
                    'target2': row[3],
                    'stance2': row[4],
                    'split': row[5]
                }
            for row in tweet_reader:
                if row:
                    tweet_dict[row[0]] = row[1]
        tweet_ids_existing = [twid for twid in label_dict if twid in tweet_dict]
        for tweet_id in tweet_ids_existing:
            label_entry = label_dict[tweet_id]
            instance1 = {
                Fields.ID: f'{tweet_id}_{label_entry["target1"].replace(" ", "-")}',
                Fields.TEXT1: label_entry['target1'],
                Fields.TEXT2: tweet_dict[tweet_id],
                Fields.LABEL_ORIGINAL: label_entry['stance1'],
                Fields.LABEL_UNIFIED: self.label_mapping[label_entry['stance1']],
                Fields.TASK: 'MultiTargetSD'
            }
            instance2 = {
                Fields.ID: f'{tweet_id}_{label_entry["target2"].replace(" ", "-")}',
                Fields.TEXT1: label_entry['target2'],
                Fields.TEXT2: tweet_dict[tweet_id],
                Fields.LABEL_ORIGINAL: label_entry['stance2'],
                Fields.LABEL_UNIFIED: self.label_mapping[label_entry['stance2']],
                Fields.TASK: 'MultiTargetSD'
            }
            if label_dict[tweet_id]['split'] == 'Train':
                train_set.append(instance1)
                train_set.append(instance2)
            elif label_dict[tweet_id]['split'] == 'Dev':
                dev_set.append(instance1)
                dev_set.append(instance2)
            elif label_dict[tweet_id]['split'] == 'Test':
                test_set.append(instance1)
                test_set.append(instance2)
            else:
                raise Exception(f'Unkown split: {label_dict[tweet_id]["split"]}')
        return train_set, dev_set, test_set


class PERSPECTRUMProcessor(PreProcessor):
    corpus_dir = 'PERSPECTRUM/'
    label_mapping = {
        'SUPPORT': LabelsUnified.PRO,
        'UNDERMINE': LabelsUnified.CON,
        'not-a-perspective': LabelsUnified.OTHER
    }

    def process(self, dev_size: float) -> None:
        f_claims = open(osjoin(self.corpus_path, 'perspectrum_with_answers_v1.0.json'))
        f_persp = open(osjoin(self.corpus_path, 'perspective_pool_v1.0.json'))
        claims = json.load(f_claims)
        perspectives = {persp['pId']: persp['text'] for persp in json.load(f_persp)}
        instances = []
        for claim in claims:
            claim_id = claim['cId']
            for persp_cluster in claim['perspectives']:
                label_orig = persp_cluster['stance_label_3']
                for persp_id in persp_cluster['pids']:
                    instances.append({
                        'claim_id': claim_id,
                        Fields.ID: self.instance_id,
                        Fields.TEXT1: claim['text'],
                        Fields.TEXT2: perspectives[persp_id],
                        Fields.LABEL_ORIGINAL: label_orig,
                        Fields.LABEL_UNIFIED: self.label_mapping[label_orig],
                        Fields.TASK: 'PERSPECTRUM'
                    })
                    self.instance_id += 1
        train_instances, dev_instances, test_instances = self._split_train_dev_test_set(instances)
        self._write_to_jsonlfile(train_instances,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(test_instances,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    def _split_train_dev_test_set(self,
                                  instances: instances_type
                                  ) -> Tuple[instances_type, instances_type, instances_type]:
        train_set = []
        dev_set = []
        test_set = []
        f_split = open(osjoin(self.corpus_path, 'dataset_split_v1.0.json'))
        split = json.load(f_split)
        for instance in instances:
            claim_id = str(instance['claim_id'])
            del instance['claim_id']
            if split[claim_id] == 'train':
                train_set.append(instance)
            elif split[claim_id] == 'dev':
                dev_set.append(instance)
            elif split[claim_id] == 'test':
                test_set.append(instance)
        return train_set, dev_set, test_set


class QQPProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/QQP/train.tsv',
        'dev': 'data/QQP/dev.tsv',
        'test': 'data/QQP/test.tsv',
    }
    file_names_out = {
        'train': 'data/QQP/train.jsonl',
        'dev': 'data/QQP/dev.jsonl',
        'test': 'data/QQP/test.jsonl',
    }
    corpus_dir = 'QQP/'
    label_mapping = {
        '0': LabelsUnified.PRO,
        '1': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        train_test_set = self._load(self.file_names_in['train'])
        train_set, test_set = self._split_train_dev_set(train_test_set, dev_size)
        dev_set = self._load(self.file_names_in['dev'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader)  # skip header
            for id_, qid1, qid2, question1, question2, is_duplicate in reader:
                instances.append({
                    Fields.ID: id_,
                    Fields.TEXT1: question1,
                    Fields.TEXT2: question2,
                    Fields.LABEL_ORIGINAL: is_duplicate,
                    Fields.LABEL_UNIFIED: self.label_mapping[is_duplicate],
                    Fields.TASK: 'QQP'
                })
        return instances


class RTEProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/RTE/train.tsv',
        'dev': 'data/RTE/dev.tsv',
        'test': 'data/RTE/test.tsv',
    }
    file_names_out = {
        'train': 'data/RTE/train.jsonl',
        'dev': 'data/RTE/dev.jsonl',
        'test': 'data/RTE/test.jsonl',
    }
    corpus_dir = 'RTE/'
    label_mapping = {
        'entailment': LabelsUnified.PRO,
        'not_entailment': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        train_test_set = self._load(self.file_names_in['train'])
        train_set, test_set = self._split_train_dev_set(train_test_set, dev_size)
        dev_set = self._load(self.file_names_in['dev'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            next(fin)
            for line in fin:
                index, sentence1, sentence2, label = line.strip('\n').split('\t')
                instances.append({
                    Fields.ID: index,
                    Fields.TEXT1: sentence1,
                    Fields.TEXT2: sentence2,
                    Fields.LABEL_ORIGINAL: label,
                    Fields.LABEL_UNIFIED: self.label_mapping[label],
                    Fields.TASK: 'RTE'
                })
        return instances


class SemEval2019Task7Processor(PreProcessor):
    """Code of this class is partly inspired/taken over from:
    https://github.com/UKPLab/mdl-stance-robustness/blob/
    68a606556f2492945be4c6623650f5bc17daa36e/data_utils/glue_utils.py#L799
    """
    corpus_dir = 'SemEval2019Task7/'
    label_mapping = {
        'support': LabelsUnified.PRO,
        'deny': LabelsUnified.CON,
        'comment': LabelsUnified.DISCUSS,
        'query': LabelsUnified.QUERY
    }

    def process(self, dev_size: float) -> None:
        train_dir = osjoin(self.corpus_path, 'rumoureval-2019-training-data')
        test_dir = osjoin(self.corpus_path, 'rumoureval-2019-test-data')
        # Load tweets
        train_dev_tweets = self._load_tweets(osjoin(train_dir, 'twitter-english'))
        test_tweets = self._load_tweets(osjoin(test_dir, 'twitter-en-test-data'))
        # train_tweets, dev_tweets = self._split_train_dev_tweets(tweets)
        # Load reddit posts
        train_reddit_posts = self._load_reddit(osjoin(train_dir, 'reddit-training-data'))
        dev_reddit_posts = self._load_reddit(osjoin(train_dir, 'reddit-dev-data'))
        test_reddit_posts = self._load_reddit(osjoin(test_dir, 'reddit-test-data'))
        # Load labels
        train_labels = self._load_labels(osjoin(train_dir, 'train-key.json'))
        dev_labels = self._load_labels(osjoin(train_dir, 'dev-key.json'))
        test_labels = self._load_labels(osjoin(self.corpus_path, 'final-eval-key.json'))
        # separate tweets/posts by split and add label

        train_instances, dev_instances, test_instances = self._split_label_instances(
            train_dev_tweets, train_reddit_posts, dev_reddit_posts, test_tweets, test_reddit_posts,
            train_labels, dev_labels, test_labels)
        self._write_to_jsonlfile(train_instances, osjoin(self.corpus_path,
                                                         self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances, osjoin(self.corpus_path,
                                                       self.file_names_out['dev']))
        self._write_to_jsonlfile(test_instances, osjoin(self.corpus_path,
                                                        self.file_names_out['test']))

    def _split_label_instances(
            self,
            train_dev_tweets: tws_type,
            train_reddit_posts: tws_type,
            dev_reddit_posts: tws_type,
            test_tweets: tws_type,
            test_reddit_posts: tws_type,
            train_labels: Dict[str, str],
            dev_labels: Dict[str, str],
            test_labels: Dict[str, str]
    ) -> Tuple[instances_type, instances_type, instances_type]:
        train_instances, dev_instances, test_instances = [], [], []
        for tweet_id in train_dev_tweets:
            if tweet_id in train_labels:
                train_instances.append(self._to_instance(tweet_id, train_dev_tweets[tweet_id],
                                                         train_labels[tweet_id]))
            elif tweet_id in dev_labels:
                dev_instances.append(self._to_instance(tweet_id, train_dev_tweets[tweet_id],
                                                       dev_labels[tweet_id]))
            else:
                raise Exception(f'Error. Tweet-id {tweet_id} no in train or dev labels.')
        for post_id in train_reddit_posts:
            train_instances.append(self._to_instance(post_id, train_reddit_posts[post_id],
                                                     train_labels[post_id]))
        for post_id in dev_reddit_posts:
            dev_instances.append(self._to_instance(post_id, dev_reddit_posts[post_id],
                                                   dev_labels[post_id]))
        for tweet_id in test_tweets:
            test_instances.append(self._to_instance(tweet_id, test_tweets[tweet_id],
                                                    test_labels[tweet_id]))
        for post_id in test_reddit_posts:
            test_instances.append(self._to_instance(post_id, test_reddit_posts[post_id],
                                                    test_labels[post_id]))
        return train_instances, dev_instances, test_instances

    def _to_instance(self, item_id: str, texts: Tuple[str, str], label: str) -> Dict[str, str]:
        return {
            Fields.ID: item_id,
            Fields.TEXT1: texts[0],
            Fields.TEXT2: texts[1],
            Fields.LABEL_ORIGINAL: label,
            Fields.LABEL_UNIFIED: self.label_mapping[label],
            Fields.TASK: 'SemEval2019Task7'
        }

    @staticmethod
    def _load_labels(fpath: str) -> Dict[str, str]:
        with open(fpath) as fin:
            return json.load(fin)['subtaskaenglish']

    @staticmethod
    def _load_tweets(dir_path: str) -> Dict[str, Tuple[str, str]]:
        """
        Returns:
            {reply_twid: (source_text, reply_text)}
        """
        # source_tweets = {}
        tweets = {}
        for topic_dir in os.listdir(dir_path):
            for thread_dir in os.listdir(osjoin(dir_path, topic_dir)):
                path_thread_dir = osjoin(dir_path, topic_dir, thread_dir)
                fname_source = os.listdir(osjoin(path_thread_dir, 'source-tweet'))[0]
                fpath_source = osjoin(path_thread_dir, 'source-tweet', fname_source)
                with open(fpath_source, encoding='ISO-8859-1') as f_source:
                    source_dict = json.load(f_source)
                    # source_tweets[source_dict['id']] = source_dict['text']
                replies_dir = osjoin(path_thread_dir, 'replies')
                for fname_reply in os.listdir(replies_dir):
                    with open(osjoin(replies_dir, fname_reply), encoding='ISO-8859-1') as f_reply:
                        reply_dict = json.load(f_reply)
                        tweets[str(reply_dict['id'])] = (source_dict['text'], reply_dict['text'])
        return tweets

    @staticmethod
    def _load_reddit(dir_path: str) -> Dict[str, Tuple[str, str]]:
        """
        Returns:
            {reply_id: (source_text, reply_text)}
        """
        posts = {}
        for thread_id in os.listdir(dir_path):
            thread_path = osjoin(dir_path, thread_id)
            fname_source = os.listdir(osjoin(thread_path, 'source-tweet'))[0]
            fpath_source = osjoin(thread_path, 'source-tweet', fname_source)
            with open(fpath_source, encoding='ISO-8859-1') as f_source:
                source_dict = json.load(f_source)
                # source_id = source_dict['data']['children'][0]['data']['id']
                source_text = source_dict['data']['children'][0]['data']['title']
            for fname_reply in os.listdir(osjoin(thread_path, 'replies')):
                fpath_reply = osjoin(thread_path, 'replies', fname_reply)
                with open(fpath_reply, encoding='ISO-8859-1') as f_reply:
                    reply_dict = json.load(f_reply)
                    if 'body' in reply_dict['data']:
                        reply_id = reply_dict['data']['id']
                        reply_text = reply_dict['data']['body']
                        posts[reply_id] = (source_text, reply_text)
        return posts


class SnopesProcessor(PreProcessor):
    file_names_in = {
        'train': 'ukp_snopes_corpus/datasets/snopes.stance.dev.jsonl',
        'dev': 'ukp_snopes_corpus/datasets/snopes.stance.test.jsonl',
        'test': 'ukp_snopes_corpus/datasets/snopes.stance.train.jsonl',
        'data': 'ukp_snopes_corpus/datasets_raw/snopes_corpus_2.csv'
    }
    corpus_dir = 'Snopes/'
    label_mapping = {
        'agree': LabelsUnified.PRO,
        'refute': LabelsUnified.CON,
        'nostance': LabelsUnified.NONE,
    }

    def process(self, dev_size: float) -> None:
        fpath_data = osjoin(self.corpus_path, self.file_names_in['data'])
        instances = self._load_instances(fpath_data)
        train_instances, dev_instances, test_instances = self._split_train_dev_test(instances)
        self._write_to_jsonlfile(train_instances,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    def _load_instances(self, fpath: str) -> instances_type:
        instances = []
        # ID,Category,Sub-Category,Snopes URL,Headline,Description,Claim,Truthfulness (Verdicts),
        # Source,Snippets (ETS),Origin,Stance,Evidence (FGE numbers)
        fin = open(fpath, encoding='utf8')
        reader = csv.reader(fin)
        next(reader)
        for row in reader:
            evid_sents = []
            row_id, claim, evid_snpts, stance, evid_ids = row[0], row[6], row[9], row[11], row[12]
            if not evid_ids:
                continue
            claim = claim.replace('“', "'").replace('”', "'")  # avoid encoding errors
            evidence_ids = evid_ids.split('_')[1:]
            results = re.findall(r'\d{1,2}_{.+?}', evid_snpts)
            for evid in results:
                evid_id = re.search(r'(^\d+)_', evid).group(1)
                evid_sent = re.search(r'{(.+)}', evid).group(1)
                if evid_id in evidence_ids:
                    evid_sents.append(evid_sent)
            instances.append({
                Fields.ID: row_id,
                Fields.TEXT1: claim.strip(),
                Fields.TEXT2: ' '.join(evid_sents),
                Fields.LABEL_ORIGINAL: stance,
                Fields.LABEL_UNIFIED: self.label_mapping[stance],
                Fields.TASK: 'Snopes'
            })
        return instances

    def _split_train_dev_test(self, instances: instances_type):
        train_ids = self._load_ids(osjoin(self.corpus_path, self.file_names_in['train']))
        dev_ids = self._load_ids(osjoin(self.corpus_path, self.file_names_in['dev']))
        test_ids = self._load_ids(osjoin(self.corpus_path, self.file_names_in['test']))
        train_instances, dev_instances, test_instances = [], [], []
        for instance in instances:
            if instance['id'] in train_ids:
                train_instances.append(instance)
            elif instance['id'] in dev_ids:
                dev_instances.append(instance)
            elif instance['id'] in test_ids:
                test_instances.append(instance)
            else:
                msg = f'Error. Instance id {instance["id"]} not found in train, dev or test set.'
                raise Exception(msg)
        return train_instances, dev_instances, test_instances

    @staticmethod
    def _load_ids(fpath: str) -> Set[str]:
        ids = set()
        with open(fpath) as fin:
            for line in fin:
                ids.add(str(json.loads(line)['id']))
        return ids


class SCDProcessor(PreProcessor):
    corpus_dir = 'SCD/'
    data_dirs = {
        'abortion': 'stance/abortion',
        'gayRights': 'stance/gayRights',
        'marijuana': 'stance/marijuana',
        'obama': 'stance/obama'
    }
    label_mapping = {
        '+1': LabelsUnified.PRO,
        '-1': LabelsUnified.CON,
    }
    data_splits = {
        'abortion': 'train',
        'gayRights': 'train',
        'marijuana': 'dev',
        'obama': 'test'
    }
    topic_to_text1 = {
        'abortion': 'abortion',
        'gayRights': 'gay rights',
        'marijuana': 'marijuana',
        'obama': 'Obama'
    }

    def process(self, dev_size: float) -> None:
        data_path = osjoin(self.corpus_path, 'stance')
        instances = self._load_instances(data_path)
        train_instances, dev_instances, test_instances = self._split_train_dev_test(instances)
        self._write_to_jsonlfile(train_instances,
                                 osjoin(self.corpus_path, self.file_names_out['train']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['dev']))
        self._write_to_jsonlfile(dev_instances,
                                 osjoin(self.corpus_path, self.file_names_out['test']))

    def _split_train_dev_test(self, instances: instances_type
                              ) -> Tuple[instances_type, instances_type, instances_type]:
        train_instances, dev_instances, test_instances = [], [], []
        for instance in instances:
            topic = instance['id'].split('_')[0]
            if self.data_splits[topic] == 'train':
                train_instances.append(instance)
            elif self.data_splits[topic] == 'dev':
                dev_instances.append(instance)
            elif self.data_splits[topic] == 'test':
                test_instances.append(instance)
            else:
                raise Exception('This should not happen.')
        return train_instances, dev_instances, test_instances

    def _load_instances(self, data_path: str) -> instances_type:
        instances = []
        stances = ['+1', '-1']
        topics = ['abortion', 'gayRights', 'marijuana', 'obama']
        for topic in topics:
            tpath = osjoin(data_path, topic)
            post_id_to_files = self._get_post_id_to_files(tpath)
            for post_id in post_id_to_files:
                with open(osjoin(tpath, post_id_to_files[post_id]['data'])) as f_data:
                    post = f_data.read().strip('\n')
                with open(osjoin(tpath, post_id_to_files[post_id]['meta'])) as f_meta:
                    stance = f_meta.readlines()[2].strip().split('=')[1]
                    if stance not in stances:
                        # The corpus contains one instance without an annotation, skip it.
                        continue
                instances.append({
                    Fields.ID: f'{topic}_{post_id}',
                    Fields.TEXT1: self.topic_to_text1[topic],
                    Fields.TEXT2: post,
                    Fields.LABEL_ORIGINAL: self.label_mapping[stance],
                    # already mapped to uni-labels because -1/+1 must be mapped
                    Fields.LABEL_UNIFIED: self.label_mapping[stance],
                    Fields.TASK: 'SCD'
                })
        return instances

    @staticmethod
    def _get_post_id_to_files(tpath: str) -> DefaultDict[str, Dict[str, str]]:
        """
        Returns:
            {comment_id: {'data': fname_text, 'meta': fname_meta})}
        """
        post_id_to_files = defaultdict(dict)
        fnames = os.listdir(tpath)
        for fname in fnames:
            id_, suffix = fname.split('.')
            post_id_to_files[id_][suffix] = fname
        assert all([len(post_id_to_files[k]) == 2 for k in post_id_to_files])
        return post_id_to_files


class SemEval2016Task4AProcessor(PreProcessor):
    file_names_in = {
        'train_labels': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                        'subtask-A.train.gold.txt',
        'train_tweets': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                        'subtask-A.train.gold.txt.tweets',
        'dev_labels': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                      'subtask-A.dev.gold.txt',
        'dev_tweets': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                      'subtask-A.dev.gold.txt.tweets',
        'test_labels': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                       'subtask-A.devtest.gold.txt',
        'test_tweets': 'data/SemEval2016Task4A/100_topics_100_tweets.sentence-three-point.'
                       'subtask-A.devtest.gold.txt.tweets'
    }
    file_names_out = {
        'train': 'data/SemEval2016Task4A/train.jsonl',
        'dev': 'data/SemEval2016Task4A/dev.jsonl',
        'test': 'data/SemEval2016Task4A/test.jsonl',
    }
    corpus_dir = 'SemEval2016Task4A/'
    label_mapping = {
        'positive': LabelsUnified.PRO,
        'negative': LabelsUnified.CON,
        'neutral': LabelsUnified.OTHER
    }

    def process(self, dev_size: float) -> None:
        train_set = self._process(self.file_names_in['train_labels'],
                                  self.file_names_in['train_tweets'])
        dev_set = self._process(self.file_names_in['dev_labels'],
                                self.file_names_in['dev_tweets'])
        test_set = self._process(self.file_names_in['test_labels'],
                                 self.file_names_in['test_tweets'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _process(self, path_annotations: str, path_tweets: str) -> instances_type:
        annotation_reader = csv.reader(open(path_annotations), delimiter='\t')
        tweet_reader = csv.reader(open(path_tweets))
        tweets = {}
        for row in tweet_reader:
            tweets[row[0]] = {Fields.ID: row[0], Fields.TEXT2: row[1]}
        for row in annotation_reader:
            if row[0] in tweets:
                tweets[row[0]][Fields.TEXT1] = NO_TARGET_TOKEN
                tweets[row[0]][Fields.LABEL_ORIGINAL] = row[1].strip()
                tweets[row[0]][Fields.LABEL_UNIFIED] = self.label_mapping[row[1].strip()]
                tweets[row[0]][Fields.TASK] = 'SemEval2016Task4A'
        return [tweet for tweet_id, tweet in tweets.items()]


class SemEval2016Task4BProcessor(PreProcessor):
    file_names_in = {
        'train_labels': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                        'subtask-BD.train.gold.txt',
        'train_tweets': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                        'subtask-BD.train.gold.txt.tweets',
        'dev_labels': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                      'subtask-BD.dev.gold.txt',
        'dev_tweets': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                      'subtask-BD.dev.gold.txt.tweets',
        'test_labels': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                       'subtask-BD.devtest.gold.txt',
        'test_tweets': 'data/SemEval2016Task4B/100_topics_XXX_tweets.topic-two-point.'
                       'subtask-BD.devtest.gold.txt.tweets'
    }
    file_names_out = {
        'train': 'data/SemEval2016Task4B/train.jsonl',
        'dev': 'data/SemEval2016Task4B/dev.jsonl',
        'test': 'data/SemEval2016Task4B/test.jsonl',
    }
    corpus_dir = 'SemEval2016Task4B/'
    label_mapping = {
        'positive': LabelsUnified.PRO,
        'negative': LabelsUnified.CON,
    }

    def process(self, dev_size: float) -> None:
        train_set = self._process(self.file_names_in['train_labels'],
                                  self.file_names_in['train_tweets'])
        dev_set = self._process(self.file_names_in['dev_labels'],
                                self.file_names_in['dev_tweets'])
        test_set = self._process(self.file_names_in['test_labels'],
                                 self.file_names_in['test_tweets'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _process(self, path_annotations: str, path_tweets: str) -> instances_type:
        annotation_reader = csv.reader(open(path_annotations), delimiter='\t')
        tweet_reader = csv.reader(open(path_tweets))
        tweets = {}
        for row in tweet_reader:
            tweets[row[0]] = {Fields.ID: row[0], Fields.TEXT2: row[1]}
        for row in annotation_reader:
            if row[0] in tweets:
                tweets[row[0]][Fields.TEXT1] = row[1].strip('@')
                tweets[row[0]][Fields.LABEL_ORIGINAL] = row[2].strip()
                tweets[row[0]][Fields.LABEL_UNIFIED] = self.label_mapping[row[2].strip()]
                tweets[row[0]][Fields.TASK] = 'SemEval2016Task4B'
        return [tweet for tweet_id, tweet in tweets.items()]


class SemEval2016Task4CProcessor(PreProcessor):
    file_names_in = {
        'train_labels': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                        'subtask-CE.train.gold.txt',
        'train_tweets': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                        'subtask-CE.train.gold.txt.tweets',
        'dev_labels': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                      'subtask-CE.dev.gold.txt',
        'dev_tweets': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                      'subtask-CE.dev.gold.txt.tweets',
        'test_labels': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                       'subtask-CE.devtest.gold.txt',
        'test_tweets': 'data/SemEval2016Task4C/100_topics_100_tweets.topic-five-point.'
                       'subtask-CE.devtest.gold.txt.tweets'
    }
    file_names_out = {
        'train': 'data/SemEval2016Task4C/train.jsonl',
        'dev': 'data/SemEval2016Task4C/dev.jsonl',
        'test': 'data/SemEval2016Task4C/test.jsonl',
    }
    corpus_dir = 'SemEval2016Task4C/'
    label_mapping = {
        '-2': LabelsUnified.CON,
        '-1': LabelsUnified.CON,
        '0': LabelsUnified.OTHER,
        '1': LabelsUnified.PRO,
        '2': LabelsUnified.PRO,
    }

    def process(self, dev_size: float) -> None:
        train_set = self._process(self.file_names_in['train_labels'],
                                  self.file_names_in['train_tweets'])
        dev_set = self._process(self.file_names_in['dev_labels'],
                                self.file_names_in['dev_tweets'])
        test_set = self._process(self.file_names_in['test_labels'],
                                 self.file_names_in['test_tweets'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _process(self, path_annotations: str, path_tweets: str) -> instances_type:
        annotation_reader = csv.reader(open(path_annotations), delimiter='\t')
        tweet_reader = csv.reader(open(path_tweets))
        tweets = {}
        for row in tweet_reader:
            tweets[row[0]] = {Fields.ID: row[0], Fields.TEXT2: row[1]}
        for row in annotation_reader:
            if row[0] in tweets:
                tweets[row[0]][Fields.TEXT1] = row[1].strip('@')
                tweets[row[0]][Fields.LABEL_ORIGINAL] = row[2].strip()
                tweets[row[0]][Fields.LABEL_UNIFIED] = self.label_mapping[row[2].strip()]
                tweets[row[0]][Fields.TASK] = 'SemEval2016Task4C'
        return [tweet for tweet_id, tweet in tweets.items()]


class SSTProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/SST/train_raw.jsonl',
        'dev': 'data/SST/dev_raw.jsonl',
        'test': 'data/SST/test_raw.jsonl',
    }
    file_names_out = {
        'train': 'data/SST/train.jsonl',
        'dev': 'data/SST/dev.jsonl',
        'test': 'data/SST/test.jsonl',
    }
    corpus_dir = 'SST/'
    label_mapping = {
        'pro': LabelsUnified.PRO,
        'con': LabelsUnified.CON,
    }

    def process(self, dev_size: float) -> None:
        self._process(self.file_names_in['train'], self.file_names_out['train'])
        self._process(self.file_names_in['dev'], self.file_names_out['dev'])
        self._process(self.file_names_in['test'], self.file_names_out['test'])

    def _process(self, fpath_in: str, fpath_out: str) -> None:
        with open(fpath_in) as fin, open(fpath_out, 'w') as fout:
            for i, line in enumerate(fin):
                instance = json.loads(line)
                bin_label = self._get_binary_label(instance)
                # fine_label = self._get_fine_label(instance)
                fout.write(json.dumps({
                    Fields.ID: i,
                    Fields.TEXT1: NO_TARGET_TOKEN,
                    Fields.TEXT2: instance['sentence'],
                    Fields.LABEL_ORIGINAL: bin_label,
                    Fields.LABEL_UNIFIED: self.label_mapping[bin_label],
                    Fields.TASK: 'SST'
                }) + '\n')

    @staticmethod
    def _get_binary_label(instance: Dict[str, Union[float, str]]) -> str:
        score = instance['label']
        if score < 0.5:
            return 'con'
        else:
            return 'pro'


class STSBProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/STSB/train.tsv',
        'dev': 'data/STSB/dev.tsv',
        'test': 'data/STSB/test.tsv',
    }
    file_names_out = {
        'train': 'data/STSB/train.jsonl',
        'dev': 'data/STSB/dev.jsonl',
        'test': 'data/STSB/test.jsonl',
    }
    corpus_dir = 'STSB/'
    label_mapping = {
        '0': LabelsUnified.CON,
        '1': LabelsUnified.CON,
        '2': LabelsUnified.OTHER,
        '3': LabelsUnified.OTHER,
        '4': LabelsUnified.PRO,
        '5': LabelsUnified.PRO,
    }

    def process(self, dev_size: float) -> None:
        train_test_set = self._load(self.file_names_in['train'])
        train_set, test_set = self._split_train_dev_set(train_test_set, dev_size)
        dev_set = self._load(self.file_names_in['dev'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            next(fin)
            for line in fin:
                idx, genre, fname, year, oldidx, src1, src2, sent1, sent2, score = line.split('\t')
                label_orig = str(int(round(float(score))))
                instances.append({
                    Fields.ID: idx,
                    Fields.TEXT1: sent1,
                    Fields.TEXT2: sent2,
                    Fields.LABEL_ORIGINAL: label_orig,
                    Fields.LABEL_UNIFIED: self.label_mapping[label_orig],
                    Fields.TASK: 'STSB'
                })
        return instances


class TargetDepSAProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/TargetDepSA/train.raw',
        'test': 'data/TargetDepSA/test.raw',
    }
    file_names_out = {
        'train': 'data/TargetDepSA/train.jsonl',
        'dev': 'data/TargetDepSA/dev.jsonl',
        'test': 'data/TargetDepSA/test.jsonl',
    }
    corpus_dir = 'TargetDepSA/'
    label_mapping = {
        '1': LabelsUnified.PRO,
        '-1': LabelsUnified.CON,
        '0': LabelsUnified.OTHER
    }

    def process(self, dev_size: float) -> None:
        train_dev_data = self._load(self.file_names_in['train'])
        train_data, dev_data = self._split_train_dev_set(train_dev_data, dev_size)
        test_data = self._load(self.file_names_in['test'])
        self._write_to_jsonlfile(train_data, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_data, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_data, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> List[Dict[str, Union[int, str]]]:
        instances = []
        with open(fpath_in) as fin:
            cur_sent = ''
            cur_target = ''
            cur_label = ''
            for i, line in enumerate(fin):
                if i % 3 == 0:
                    if cur_sent:
                        instances.append({
                            Fields.ID: int(i / 3),
                            Fields.TEXT1: cur_target,
                            Fields.TEXT2: cur_sent.replace('$T$', cur_target),
                            Fields.LABEL_ORIGINAL: cur_label,
                            Fields.LABEL_UNIFIED: self.label_mapping[cur_label],
                            Fields.TASK: 'TargetDepSA'
                        })
                    cur_sent = line.strip()
                elif (i + 2) % 3 == 0:
                    cur_target = line.strip()
                elif (i + 1) % 3 == 0:
                    cur_label = line.strip()
        return instances


class WNLIProcessor(PreProcessor):
    file_names_in = {
        'train': 'data/WNLI/train.tsv',
        'dev': 'data/WNLI/dev.tsv',
        'test': 'data/WNLI/test.tsv',
    }
    file_names_out = {
        'train': 'data/WNLI/train.jsonl',
        'dev': 'data/WNLI/dev.jsonl',
        'test': 'data/WNLI/test.jsonl',
    }
    corpus_dir = 'WNLI/'
    label_mapping = {
        '1': LabelsUnified.PRO,
        '0': LabelsUnified.CON
    }

    def process(self, dev_size: float) -> None:
        train_test_set = self._load(self.file_names_in['train'])
        train_set, test_set = self._split_train_dev_set(train_test_set, dev_size)
        dev_set = self._load(self.file_names_in['dev'])
        self._write_to_jsonlfile(train_set, self.file_names_out['train'])
        self._write_to_jsonlfile(dev_set, self.file_names_out['dev'])
        self._write_to_jsonlfile(test_set, self.file_names_out['test'])

    def _load(self, fpath_in: str) -> instances_type:
        instances = []
        with open(fpath_in) as fin:
            next(fin)
            for line in fin:
                index, sentence1, sentence2, label = line.strip('\n').split('\t')
                instances.append({
                    Fields.ID: index,
                    Fields.TEXT1: sentence1,
                    Fields.TEXT2: sentence2,
                    Fields.LABEL_ORIGINAL: label,
                    Fields.LABEL_UNIFIED: self.label_mapping[label],
                    Fields.TASK: 'WNLI'
                })
        return instances


def main(args: argparse.Namespace):
    print(f'Start processing {args.corpus}.')
    print(f'  data-dir: {args.data_dir}')
    print(f'  dev-size: {args.size_dev}')
    processor = CORPUS_NAME_TO_PROCESSOR[args.corpus]
    print(f'  Use processor: {processor.__name__}')
    processor(args.data_dir).process(args.size_dev)
    print(f'Finished processing {args.corpus}.')


CORPUS_NAME_TO_PROCESSOR = {
    'arc': ArcProcessor,
    'ArgMin': ArgMinProcessor,
    'CoLA': CoLAProcessor,
    'FNC1': FNC1Processor,
    'IAC': IACProcessor,
    'IBMCS': IBMCSProcessor,
    'IMDB': IMDBProcessor,
    'ISarcasm': ISarcasmProcessor,
    'MSRPara': MSRParaProcessor,
    'MultiNLI': MultiNLIProcessor,
    'MultiTargetSD': MultiTargetSDProcessor,
    'PERSPECTRUM': PERSPECTRUMProcessor,
    'QQP': QQPProcessor,
    'RTE': RTEProcessor,
    'SCD': SCDProcessor,
    'SemEval2016Task4A': SemEval2016Task4AProcessor,
    'SemEval2016Task4B': SemEval2016Task4BProcessor,
    'SemEval2016Task4C': SemEval2016Task4CProcessor,
    'SemEval2016Task6': SemEval2016Task6Processor,
    'SemEval2019Task7': SemEval2019Task7Processor,
    'Snopes': SnopesProcessor,
    'SST': SSTProcessor,
    'STSB': STSBProcessor,
    'TargetDepSA': TargetDepSAProcessor,
    'WNLI': WNLIProcessor
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, choices=list(CORPUS_NAME_TO_PROCESSOR.keys()),
                        help='Name of corpus to process.')
    parser.add_argument('-s', '--size_dev', type=float,
                        help='Size of devset as proportion of trainset.')
    parser.add_argument('-d', '--data_dir', type=str,
                        help='Path to directory containing datasets.')
    cmd_args = parser.parse_args()
    main(cmd_args)
