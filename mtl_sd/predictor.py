import argparse
from copy import deepcopy
from typing import List, Dict, Optional, Any, Type

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from overrides import overrides

from allennlp.data import Instance
from allennlp.predictors import MultiTaskPredictor
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import MetadataField
from allennlp.common.util import JsonDict, sanitize


@Predictor.register('head_predictor')
class HeadPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        target = json_dict['text1']
        claim = json_dict['text2']
        return self._dataset_reader.text_to_instance(target, claim)


@Predictor.register("multitask_stance")
class MultiTaskStancePredictor(MultiTaskPredictor):
    """Subclass MultiTaskPredictor to prevent encoder outputs being
    written to output files.
    """

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        task_field = instance["task"]
        if not isinstance(task_field, MetadataField):
            raise ValueError(self._WRONG_FIELD_ERROR)
        task: str = task_field.metadata
        if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
            raise ConfigurationError(self._WRONG_READER_ERROR)
        self._dataset_reader.readers[task].apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        # delete encoder outputs
        del outputs['token_ids_encoded']
        del outputs['token_ids']
        del outputs['mask']
        del outputs['type_ids']
        return sanitize(outputs)


@Predictor.register("multitask_stance_regression")
class MultiTaskStanceRegressionPredictor(MultiTaskStancePredictor):
    """Subclass MultiTaskPredictor to prevent encoder outputs being
    written to output files.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        task = json_dict["task"]
        if list(self.predictors.keys())[0] == 'UNIFIED_regr':
            task = 'UNIFIED'
        del json_dict["task"]
        task += '_regr'
        predictor = self.predictors[task]
        instance = predictor._json_to_instance(json_dict)
        instance.add_field("task", MetadataField(task))
        return instance


@Predictor.register("multitask_stance_classification")
class MultiTaskStanceClassificationPredictor(MultiTaskStancePredictor):
    """Subclass MultiTaskPredictor to prevent encoder outputs being
    written to output files.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        task = json_dict["task"]
        if list(self.predictors.keys())[0] == 'UNIFIED_class':
            task = 'UNIFIED'
        del json_dict["task"]
        task += '_class'
        predictor = self.predictors[task]
        instance = predictor._json_to_instance(json_dict)
        instance.add_field("task", MetadataField(task))
        return instance
