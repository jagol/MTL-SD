import argparse
from copy import deepcopy
from typing import List, Dict, Optional, Any, Type

from allennlp.common.checks import check_for_gpu
from allennlp.models import Archive, load_archive
from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import MultiTaskPredictor
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.data.fields import LabelField

# from mtl_sd.dataset_reader import DATASET_TO_READER


@Predictor.register('head_predictor')
class HeadPredictor(Predictor):

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({'sentence': sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        target = json_dict['text1']
        claim = json_dict['text2']
        return self._dataset_reader.text_to_instance(target, claim)

#     @classmethod
#     def from_archive(
#             cls,
#             archive: Archive,
#             dataset_name: str,
#             predictor_name: str = None,
#             dataset_reader_to_load: str = "validation",
#             frozen: bool = True,
#             extra_args: Optional[Dict[str, Any]] = None,
#     ) -> "Predictor":
#         """
#         Same as the original from_archive-method (https://github.com/allenai/allennlp/blob/
#         d2ae540d489336ba05f15479d3c55530b0bd6949/allennlp/predictors/predictor.py#L368)
#         but adjusted for mtl, by loading a dataset-specific dataset-reader.
#
#         Instantiate a `Predictor` from an [`Archive`](../models/archival.md);
#         that is, from the result of training a model. Optionally specify which `Predictor`
#         subclass; otherwise, we try to find a corresponding predictor in `DEFAULT_PREDICTORS`,
#         or if one is not found, the base class (i.e. `Predictor`) will be used. Optionally specify
#         which [`DatasetReader`](../data/dataset_readers/dataset_reader.md) should be loaded;
#         otherwise, the validation one will be used if it exists followed by the training dataset
#         reader. Optionally specify if the loaded model should be frozen, meaning `model.eval()`
#         will be called.
#         """
#         # Duplicate the config so that the config inside the archive doesn't get consumed
#         config = archive.config.duplicate()
#
#         if not predictor_name:
#             model_type = config.get("model").get("type")
#             model_class, _ = Model.resolve_class_name(model_type)
#             predictor_name = model_class.default_predictor
#         predictor_class: Type[Predictor] = (
#             Predictor.by_name(predictor_name) if predictor_name is not None else cls
#         # type: ignore
#         )
#
#         dataset_reader = DATASET_TO_READER[dataset_name]
#
#         if dataset_reader_to_load == "validation":
#             dataset_reader = archive.validation_dataset_reader
#         else:
#             dataset_reader = archive.dataset_reader
#
#         model = archive.model
#         if frozen:
#             model.eval()
#
#         if extra_args is None:
#             extra_args = {}
#
#         return predictor_class(model, dataset_reader, **extra_args)
#
#
# def _get_predictor(args: argparse.Namespace) -> Predictor:
#     check_for_gpu(args.cuda_device)
#     archive = load_archive(
#         args.archive_file,
#         weights_file=args.weights_file,
#         cuda_device=args.cuda_device,
#         overrides=args.overrides,
#     )
#
#     predictor_args = args.predictor_args.strip()
#     if len(predictor_args) <= 0:
#         predictor_args = {}
#     else:
#         import json
#
#         predictor_args = json.loads(predictor_args)
#
#     return Predictor.from_archive(
#         archive,
#         dataset_name=,
#         args.predictor,
#         dataset_reader_to_load=args.dataset_reader_choice,
#         extra_args=predictor_args,
#     )
