import abc
import math
import os
import tempfile
import warnings

import numpy as np
import torch

from mmlib.equal import tensor_equal
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.save_info import SingleModelSaveInfo, ProvSingleModelSaveInfo, ModelListSaveInfo
from mmlib.schema.dataset import Dataset
from mmlib.schema.file_reference import FileReference
from mmlib.schema.model_info import ModelInfo, MODEL_INFO
from mmlib.schema.model_list_info import ModelListInfo
from mmlib.schema.recover_info import FullModelRecoverInfo, WeightsUpdateRecoverInfo, ProvenanceRecoverInfo, \
    FullModelListRecoverInfo, CompressedModelListRecoverInfo
from mmlib.schema.restorable_object import RestoredModelInfo, RestoredModelListInfo
from mmlib.schema.store_type import ModelStoreType, ModelListStoreType
from mmlib.schema.train_info import TrainInfo
from mmlib.track_env import compare_env_to_current
from mmlib.util.helper import log_start, log_stop, torch_dtype_to_numpy_dict, to_tensor
from mmlib.util.init_from_file import create_object, create_type
from mmlib.util.weight_dict_merkle_tree import WeightDictMerkleTree, THIS, OTHER

BASELINE = 'baseline'
PARAM_UPDATE = 'param_update'
PROVENANCE = 'provenance'

START = 'START'
STOP = 'STOP'

PICKLED_MODEL_PARAMETERS = 'pickled_model_parameters'
PARAMETERS_PATCH = "parameters_patch"
RESTORE_PATH = 'restore_path'
MODEL_WEIGHTS = 'model_weights.pt'
PARAMETERS = 'parameters'

COMPRESS_FUNC = 'compress_func'
DECOMPRESS_FUNC = 'decompress_func'
COMPRESS_KWARGS = 'compress_kwargs'
DECOMPRESS_KWARGS = 'decompress_kwargs'


# Future work, se if it would make sense to use protocol here
class AbstractSaveService(metaclass=abc.ABCMeta):

    def __init__(self, logging=False):
        """
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        self.logging = logging

    @abc.abstractmethod
    def save_model(self, model_save_info: SingleModelSaveInfo) -> str:
        """
        Saves a model together with the given metadata.
        :param model_save_info: An instance of ModelSaveInfo providing all the info needed to save the model.
         process. - If set, time consumption for save process might rise.
        :return: Returns the id that was used to store the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        """
        Recovers a the model and metadata identified by the given model id.
        :param model_id: The id to identify the model with.
        :param execute_checks: Indicates if additional checks should be performed to ensure a correct recovery of
        the model.
        :return: The recovered model and metadata bundled in an object of type ModelRestoreInfo.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def model_save_size(self, model_id: str) -> dict:
        """
        Gives detailed information about the storage consumption of a model.
        :param model_id: The id to identify the model.
        :return: Detailed information about the storage consumption of a model -- size in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_model_ids(self) -> [str]:
        """
        Retuns a list of all stored model_ids
        """
        raise NotImplementedError


class BaselineSaveService(AbstractSaveService):
    """A Service that offers functionality to store PyTorch models by making use of a persistence service.
         The metadata is stored in JSON like dictionaries, files and weights are stored as files."""

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(logging)
        self._file_pers_service = file_pers_service
        self._dict_pers_service = dict_pers_service
        self._file_pers_service.logging = logging
        self._dict_pers_service.logging = logging

    def save_model(self, model_save_info: SingleModelSaveInfo) -> str:
        log_all = log_start(self.logging, BASELINE, 'call_save_full_model', 'all')

        self._check_consistency(model_save_info)

        # usually we would consider at this bit how we best store the given model
        # but since this is the baseline service we just store the full model every time.
        model_id = self._save_full_model(model_save_info)

        log_stop(self.logging, log_all)

        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        # in this baseline approach we always store the full model (pickled weights + code)
        log_all = log_start(self.logging, BASELINE, 'recover_model-{}'.format(model_id), 'all')
        with tempfile.TemporaryDirectory() as tmp_path:
            log_load = log_start(self.logging, BASELINE, 'recover_model', 'load_model_info_rec_files')
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)
            log_stop(self.logging, log_load)

            log_recover = log_start(self.logging, BASELINE, 'recover_model', 'recover_from_info')
            # recover model form info
            recover_info: FullModelRecoverInfo = model_info.recover_info

            model = create_object(recover_info.model_code.path, recover_info.model_class_name)
            s_dict = self._recover_pickled_weights(recover_info.parameters_file.path)
            model.load_state_dict(s_dict)

            restored_model_info = RestoredModelInfo(model=model)

            if execute_checks:
                self._check_weights(model, model_info)
                self._check_env(model_info)

        log_stop(self.logging, log_recover)
        log_stop(self.logging, log_all)
        return restored_model_info

    def model_save_size(self, model_id: str) -> dict:
        place_holder = ModelInfo.load_placeholder(model_id)
        size_dict = place_holder.size_info(self._file_pers_service, self._dict_pers_service)

        return size_dict

    def all_model_ids(self) -> [str]:
        return self._dict_pers_service.all_ids_for_type(MODEL_INFO)

    def _check_consistency(self, model_save_info):
        # when storing a full model we need the following information
        # the model itself
        assert model_save_info.model, 'model is not set'
        # the model code
        assert model_save_info.model_code, 'model code is not set'
        # the class name of the model
        assert model_save_info.model_class_name, 'model class name is not set'

    def _save_full_model(self, model_save_info: SingleModelSaveInfo, add_weights_hash_info=True) -> str:
        log_all = log_start(self.logging, BASELINE, '_save_full_model', 'all')

        with tempfile.TemporaryDirectory() as tmp_path:
            log_pickle = log_start(self.logging, BASELINE, '_save_full_model', 'pickle_weights')
            weights_path = self._pickle_weights(model_save_info.model, tmp_path)
            log_stop(self.logging, log_pickle)

            base_model = model_save_info.base_model if model_save_info.base_model else None

            # models are recovered in a tmp directory and only the model object is returned
            # this is why the inferred model code path might not exists anymore, we have to check this
            # and if it is not existing anymore, we have to restore the code for the base model

            if not os.path.isfile(model_save_info.model_code):
                assert base_model, 'code not given and no base model'
                base_model_info = ModelInfo.load(base_model, self._file_pers_service, self._dict_pers_service, tmp_path)
                model_code = self._restore_code_from_base_model(base_model_info, tmp_path)
                model_save_info.model_code = model_code.path

            recover_info = FullModelRecoverInfo(parameters_file=FileReference(path=weights_path),
                                                model_code=FileReference(path=model_save_info.model_code),
                                                model_class_name=model_save_info.model_class_name,
                                                environment=model_save_info.environment)

            log_weight_hash = log_start(self.logging, BASELINE, '_save_full_model', '_get_weights_hash_info')
            weights_hash_info = _get_weights_hash_info(add_weights_hash_info, model_save_info)
            log_stop(self.logging, log_weight_hash)

            model_info = ModelInfo(store_type=ModelStoreType.FULL_MODEL, recover_info=recover_info,
                                   derived_from_id=base_model, weights_hash_info=weights_hash_info)

            log_persist = log_start(self.logging, BASELINE, '_save_full_model', 'persist_model_info')
            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
            log_stop(self.logging, log_persist)

            log_stop(self.logging, log_all)

            return model_info_id

    def _restore_code_from_base_model(self, model_info: ModelInfo, tmp_path):
        assert isinstance(model_info, ModelInfo)

        code, _ = self._restore_code_and_class_name(model_info, tmp_path)
        return code

    def _find_nearest_full_model_info(self, model_info, restore_dir):
        current_model_info = model_info
        while not (hasattr(current_model_info, 'store_type') and
                   current_model_info.store_type == ModelStoreType.FULL_MODEL):
            base_model_id = current_model_info.derived_from
            base_model_info = ModelInfo.load(
                obj_id=base_model_id,
                file_pers_service=self._file_pers_service,
                dict_pers_service=self._dict_pers_service,
                restore_root=restore_dir,
            )
            current_model_info = base_model_info
        full_model_info: ModelInfo = current_model_info
        return full_model_info

    def _restore_code_and_class_name(self, model_info: ModelInfo, tmp_path):
        full_model_info = self._find_nearest_full_model_info(model_info, tmp_path)
        assert isinstance(full_model_info.recover_info, FullModelRecoverInfo), 'model info has to be full model info'
        recover_info: FullModelRecoverInfo = full_model_info.recover_info
        # make sure all required fields are loaded
        if not (recover_info.model_class_name and recover_info.model_code):
            recover_info.load_all_fields(self._file_pers_service, self._dict_pers_service, tmp_path,
                                         load_recursive=True, load_files=False)
        class_name = recover_info.model_class_name
        code: FileReference = recover_info.model_code
        self._file_pers_service.recover_file(code, tmp_path)

        return code, class_name

    def _pickle_weights(self, model, save_path, model_name=None):
        # store pickle dump of model weights
        state_dict = model.state_dict()
        weight_path = self._pickle_state_dict(state_dict, save_path, model_name)

        return weight_path

    def _pickle_state_dict(self, state_dict, save_path, model_name=None):
        if model_name:
            weight_path = os.path.join(save_path, model_name)
        else:
            weight_path = os.path.join(save_path, MODEL_WEIGHTS)
        torch.save(state_dict, weight_path)
        return weight_path

    def _recover_pickled_weights(self, weights_file):
        state_dict = torch.load(weights_file)

        return state_dict

    def _get_store_type(self, model_id: str):
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
            return model_info.store_type

    def _get_base_model(self, model_id: str):
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
            return model_info.derived_from

    def _check_weights(self, model, model_info):
        log_check_weights = log_start(
            self.logging, BASELINE, '_check_weights', '_all')
        if not model_info.weights_hash_info:
            warnings.warn('no weights_hash_info available for this models')
        restored_merkle_tree: WeightDictMerkleTree = model_info.weights_hash_info
        model_merkle_tree = WeightDictMerkleTree.from_state_dict(model.state_dict())
        # NOTE maybe replace assert by throwing exception
        assert restored_merkle_tree == model_merkle_tree, 'The recovered model differs from the model that was stored'
        log_stop(self.logging, log_check_weights)

    def _check_env(self, model_info):
        # check environment
        log_check_env = log_start(
            self.logging, BASELINE, '_check_env', '_all')
        recover_info = model_info.recover_info
        envs_match = compare_env_to_current(recover_info.environment)
        assert envs_match, \
            'The current environment and the environment that was used to when storing the model differ'
        log_stop(self.logging, log_check_env)


class WeightUpdateSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 improved_version=True, logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(file_pers_service, dict_pers_service, logging)
        self.improved_version = improved_version

    def save_model(self, model_save_info: SingleModelSaveInfo) -> str:

        # as a first step we have to find out if we have to store a full model first or if we can store only the update
        # if there is no base model given, we can not compute any updates -> we have to sore the full model
        log_all = log_start(self.logging, PARAM_UPDATE, 'save_model', 'all')
        if not self._base_model_given(model_save_info):
            model_id = super().save_model(model_save_info)
        else:
            # if there is a base model, we can store the update and for a restore refer to the base model
            model_id = self._save_updated_model(model_save_info)

        log_stop(self.logging, log_all)
        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:

        log_all = log_start(self.logging, PARAM_UPDATE, 'recover_model-{}'.format(model_id), 'all')
        store_type = self._get_store_type(model_id)

        if store_type == ModelStoreType.FULL_MODEL:
            model = super().recover_model(model_id)
        else:
            model = self._recover_from_weight_update(model_id, execute_checks)

        log_stop(self.logging, log_all)
        return model

    def _recover_from_weight_update(self, model_id, execute_checks):
        log_update = log_start(self.logging, PARAM_UPDATE, 'recover_model', '_recover_from_weight_update')
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                        load_recursive=True, load_files=True)

            recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

            if recover_info.update_type == PICKLED_MODEL_PARAMETERS:
                recovered_model = self._recover_from_full_weights(model_info, tmp_path)
            elif recover_info.update_type == PARAMETERS_PATCH:
                recovered_model = self._recover_from_parameter_patch(model_info)
            else:
                raise NotImplementedError

            restored_model_info = RestoredModelInfo(model=recovered_model)

            if execute_checks:
                log_check_weights = log_start(
                    self.logging, PARAM_UPDATE, '_recover_from_weight_update', '_check_weights')
                self._check_weights(recovered_model, model_info)
                log_stop(self.logging, log_check_weights)

        log_stop(self.logging, log_update)
        return restored_model_info

    def _recover_from_full_weights(self, model_info, tmp_path):
        log = log_start(self.logging, PARAM_UPDATE, '_recover_from_full_weights', 'all')
        model_code, model_class_name = self._restore_code_and_class_name(model_info, tmp_path)
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info

        model = create_object(model_code.path, model_class_name)
        s_dict = self._recover_pickled_weights(recover_info.update.path)
        model.load_state_dict(s_dict)
        log_stop(self.logging, log)

        return model

    def _recover_from_parameter_patch(self, model_info):
        log = log_start(self.logging, PARAM_UPDATE, '_recover_from_parameter_patch', 'all')
        recover_info: WeightsUpdateRecoverInfo = model_info.recover_info
        base_model_info = self.recover_model(model_info.derived_from)
        base_model = base_model_info.model
        weights_patch = torch.load(recover_info.update.path)
        self._apply_weight_patch(base_model, weights_patch)
        log_stop(self.logging, log)

        return base_model

    def _save_updated_model(self, model_save_info, add_weights_hash_info=True):
        log_all = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'all')

        base_model_id = model_save_info.base_model
        assert base_model_id, 'no base model given'

        with tempfile.TemporaryDirectory() as tmp_path:
            log_weights_hash = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'get_weights_hash_info')
            weights_hash_info = _get_weights_hash_info(add_weights_hash_info, model_save_info)
            log_stop(self.logging, log_weights_hash)

            log_gen_update = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'generate_weights_update')
            weights_update, update_type = \
                self._generate_weights_update(model_save_info, base_model_id, weights_hash_info, tmp_path)
            log_stop(self.logging, log_gen_update)

            recover_info = WeightsUpdateRecoverInfo(update=FileReference(path=weights_update), update_type=update_type)

            model_info = ModelInfo(store_type=ModelStoreType.WEIGHT_UPDATES, recover_info=recover_info,
                                   derived_from_id=base_model_id, weights_hash_info=weights_hash_info)

            log_persist = log_start(self.logging, PARAM_UPDATE, '_save_updated_model', 'persist')
            model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
            log_stop(self.logging, log_persist)

            log_stop(self.logging, log_all)
            return model_info_id

    def _base_model_given(self, model_save_info):
        return model_save_info.base_model is not None

    def _generate_weights_update(self, model_save_info, base_model_id, weights_hash_info, tmp_path):
        base_model_info = ModelInfo.load(base_model_id, self._file_pers_service, self._dict_pers_service, tmp_path)
        current_model_weights = model_save_info.model.state_dict()

        update_save_potential = False

        if self.improved_version and base_model_info.weights_hash_info:
            diff_weights, diff_nodes = base_model_info.weights_hash_info.diff(weights_hash_info)
            assert len(diff_nodes[THIS]) == 0 and len(diff_nodes[OTHER]) == 0, \
                'models with different architecture not supported for now'

            weights_patch = current_model_weights.copy()
            # delete all keys that are the same, meaning not in the diff list
            for key in current_model_weights.keys():
                if key not in diff_weights:
                    del weights_patch[key]
        else:
            print('recover base models')
            # if there is no weights hash info given we have to fall back and load the base models
            base_model_info = self.recover_model(base_model_id)
            base_model_weights = base_model_info.model.state_dict()
            current_model_weights = model_save_info.model.state_dict()

            weights_patch = self._state_dict_patch(base_model_weights, current_model_weights)

        model_weights = super()._pickle_state_dict(weights_patch, tmp_path)
        return model_weights, PARAMETERS_PATCH

    def _state_dict_patch(self, base_model_weights, current_model_weights):
        assert base_model_weights.keys() == current_model_weights.keys(), 'given state dicts are not compatible'
        for k in list(current_model_weights.keys()):
            if tensor_equal(base_model_weights[k], current_model_weights[k]):
                del current_model_weights[k]

        return current_model_weights

    def _apply_weight_patch(self, base_model: torch.nn.Module, weights_patch):
        patched_state_dict = base_model.state_dict()
        for k, patch in weights_patch.items():
            patched_state_dict[k] = patch

        return base_model.load_state_dict(patched_state_dict)

    def _execute_checks(self, model: torch.nn.Module, model_info: ModelInfo):
        self._check_weights(model, model_info)


class ProvenanceSaveService(BaselineSaveService):

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 logging=False):
        """
        :param file_pers_service: An instance of FilePersistenceService that is used to store files.
        :param dict_pers_service: An instance of DictPersistenceService that is used to store metadata as dicts.
        :param logging: Flag that indicates if logging is turned in for this service.
        """
        super().__init__(file_pers_service, dict_pers_service, logging)

    def save_model(self, model_save_info: SingleModelSaveInfo) -> str:
        log_all = log_start(self.logging, PROVENANCE, '_save_model', 'all')
        if model_save_info.base_model is None or not isinstance(model_save_info, ProvSingleModelSaveInfo):
            # if the base model is none or model save info does not provide provenance save info we have to store the
            # model as a full model
            model_id = super().save_model(model_save_info)
        else:
            model_id = self._save_provenance_model(model_save_info)

        log_stop(self.logging, log_all)
        return model_id

    def recover_model(self, model_id: str, execute_checks: bool = True) -> RestoredModelInfo:
        log_all = log_start(self.logging, PROVENANCE, 'recover_model-{}'.format(model_id), 'all')

        base_model_id = self._get_base_model(model_id)
        if self._get_store_type(model_id) == ModelStoreType.FULL_MODEL:
            result = super().recover_model(model_id, execute_checks)
        else:
            # if there is a base model we first have to restore the base model to continue training base on it
            log_rec_base = log_start(self.logging, PROVENANCE, 'recover_model', 'recover_base_model')

            base_model_store_type = self._get_store_type(base_model_id)
            base_model_info = self._recover_base_model(base_model_id, base_model_store_type, execute_checks)
            base_model = base_model_info.model
            log_stop(self.logging, log_rec_base)

            log_load_info = log_start(self.logging, PROVENANCE, 'recover_model', 'load_model_info')
            with tempfile.TemporaryDirectory() as tmp_path:
                # TODO maybe can be replaced when using FileRef Object
                restore_dir = os.path.join(tmp_path, RESTORE_PATH)
                os.mkdir(restore_dir)

                model_info = ModelInfo.load(model_id, self._file_pers_service, self._dict_pers_service, restore_dir,
                                            load_recursive=True, load_files=True)
                recover_info: ProvenanceRecoverInfo = model_info.recover_info
                log_stop(self.logging, log_load_info)

                log_train = log_start(self.logging, PROVENANCE, 'recover_model', 'train')
                train_service = recover_info.train_info.train_service_wrapper.instance
                train_kwargs = recover_info.train_info.train_kwargs
                train_service.train(base_model, **train_kwargs)
                log_stop(self.logging, log_train)

                # because we trained it here the base_model is the updated version
                restored_model = base_model
                restored_model_info = RestoredModelInfo(model=restored_model)

                if execute_checks:
                    self._check_weights(restored_model, model_info)
                    self._check_env(model_info)

                result = restored_model_info

        log_stop(self.logging, log_all)
        return result

    def _save_provenance_model(self, model_save_info):
        log_all = log_start(self.logging, PROVENANCE, '_save_provenance_model', 'all')

        log_build_prov = log_start(self.logging, PROVENANCE, '_save_provenance_model', '_build_prov_model_info')
        model_info = self._build_prov_model_info(model_save_info)
        log_stop(self.logging, log_build_prov)

        log_persist = log_start(self.logging, PROVENANCE, '_save_provenance_model', 'persist')
        model_info_id = model_info.persist(self._file_pers_service, self._dict_pers_service)
        log_stop(self.logging, log_persist)

        log_stop(self.logging, log_all)
        return model_info_id

    def add_weights_hash_info(self, model_id: str, model: torch.nn.Module):
        model_info = ModelInfo.load_placeholder(model_id)
        weights_hash_info = WeightDictMerkleTree.from_state_dict(model.state_dict())

        model_info.add_and_persist_weights_hash_info(weights_hash_info, self._dict_pers_service)

    def _build_prov_model_info(self, model_save_info):
        tw_class_name = model_save_info.train_info.train_wrapper_class_name
        tw_code = FileReference(path=model_save_info.train_info.train_wrapper_code)
        type_ = create_type(code=tw_code.path, type_name=tw_class_name)
        train_service_wrapper = type_(
            instance=model_save_info.train_info.train_service
        )
        dataset = Dataset(FileReference(path=model_save_info.raw_dataset))
        train_info = TrainInfo(
            ts_wrapper=train_service_wrapper,
            ts_wrapper_code=tw_code,
            ts_wrapper_class_name=tw_class_name,
            train_kwargs=model_save_info.train_info.train_kwargs,
        )
        prov_recover_info = ProvenanceRecoverInfo(
            dataset=dataset,
            train_info=train_info,
            environment=model_save_info.environment
        )
        derived_from = model_save_info.base_model if model_save_info.base_model else None
        model_info = ModelInfo(store_type=ModelStoreType.PROVENANCE, recover_info=prov_recover_info,
                               derived_from_id=derived_from)
        return model_info

    def _recover_base_model(self, base_model_id, base_model_store_type, execute_checks=True):
        if base_model_store_type == ModelStoreType.FULL_MODEL:
            return super().recover_model(model_id=base_model_id, execute_checks=execute_checks)
        elif base_model_store_type == ModelStoreType.PROVENANCE:
            return self.recover_model(model_id=base_model_id, execute_checks=execute_checks)
        else:
            raise NotImplementedError


class AbstractModelListSaveService(BaselineSaveService):
    def save_model(self, model_save_info: ModelListSaveInfo) -> str:
        assert self._same_architecture(model_save_info.models), "models in model list have to have same architecture"

    def _same_architecture(self, models):
        # if only one model in list it has the same architecture
        if len(models) == 1:
            return True
        else:
            # else check if we find the same tensor shape for all keys
            first_model_state = models[0].state_dict()
            for model in models:
                current_state = model.state_dict()
                for first_model_key, first_model_tensor in first_model_state.items():
                    first_shape = first_model_tensor.shape

                    # check if layer exists
                    if first_model_key not in current_state:
                        return False

                    # if layer exists check if shape is equal
                    current_shape = current_state[first_model_key].shape
                    if not current_shape == first_shape:
                        return False

            return True


class FullModelListSaveService(AbstractModelListSaveService):
    def save_models(self, save_info: ModelListSaveInfo):
        super().save_model(model_save_info=save_info)
        models_id = self._save_full_models(save_info)

        return models_id

    def _save_full_models(self, save_info: ModelListSaveInfo):
        model_list = save_info.models

        with tempfile.TemporaryDirectory() as tmp_path:
            saved_parameter_paths = []
            for i, model in enumerate(model_list):
                model_name = f'model-{i}.pt'
                param_path = self._pickle_weights(model, tmp_path, model_name)
                saved_parameter_paths.append(FileReference(path=param_path))

            recover_info = FullModelListRecoverInfo(model_code=FileReference(path=save_info.model_code),
                                                    model_class_name=save_info.model_class_name,
                                                    environment=save_info.environment,
                                                    parameter_files=saved_parameter_paths)

            model_list_info = ModelListInfo(store_type=ModelListStoreType.FULL_MODEL, recover_info=recover_info)

            model_info_id = model_list_info.persist(self._file_pers_service, self._dict_pers_service)

            return model_info_id

    def recover_models(self, model_list_id: str, execute_checks: bool = True) -> RestoredModelListInfo:
        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelListInfo.load(model_list_id, self._file_pers_service, self._dict_pers_service, tmp_path,
                                            load_recursive=True, load_files=True)

            recover_info: FullModelListRecoverInfo = model_info.recover_info

            recovered_models = []
            for param_file in recover_info.parameter_files:
                model = create_object(recover_info.model_code.path, recover_info.model_class_name)
                s_dict = self._recover_pickled_weights(param_file.path)
                model.load_state_dict(s_dict)
                recovered_models.append(model)

            return RestoredModelListInfo(models=recovered_models)


class CompressedModelListSaveService(AbstractModelListSaveService):

    def __init__(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                 compression_info: dict = None):

        super().__init__(file_pers_service, dict_pers_service)

        if compression_info:
            # check if compression_funcs are there
            assert COMPRESS_FUNC in compression_info, 'no compression method found'
            assert DECOMPRESS_FUNC in compression_info, 'no decompression method found'
            assert COMPRESS_KWARGS in compression_info, 'no compression args found'
            assert DECOMPRESS_KWARGS in compression_info, 'no decompression args found'
            self.compression_info = compression_info
        else:
            self.compression_info = {
                COMPRESS_FUNC: None,
                COMPRESS_KWARGS: {},
                DECOMPRESS_FUNC: None,
                DECOMPRESS_KWARGS: {}
            }

    def save_models(self, save_info: ModelListSaveInfo):
        super().save_model(model_save_info=save_info)
        model_ids = self._save_compressed_models(save_info)

        return model_ids

    def model_save_size(self, model_id: str) -> dict:
        place_holder = ModelListInfo.load_placeholder(model_id)
        size_dict = place_holder.size_info(self._file_pers_service, self._dict_pers_service)

        return size_dict

    def _save_compressed_models(self, save_info):
        model_list = save_info.models

        aggregated_parameters = bytearray()
        for model in model_list:
            model_state = model.state_dict()
            for _, tensor, in model_state.items():
                aggregated_parameters.extend(tensor.numpy().tobytes())

        with tempfile.TemporaryDirectory() as tmp_path:

            compression_method = self.compression_info[COMPRESS_FUNC]
            compression_kwargs = self.compression_info[COMPRESS_KWARGS]

            # compress bytearray if method given
            if compression_method:
                aggregated_parameters = compression_method(aggregated_parameters, **compression_kwargs)

            # save aggregated parameters to binary file -> minimal overhead
            param_file = FileReference(path=os.path.join(tmp_path, PARAMETERS))

            with open(param_file.path, 'wb') as f:
                f.write(aggregated_parameters)

            recover_info = CompressedModelListRecoverInfo(model_code=FileReference(path=save_info.model_code),
                                                          model_class_name=save_info.model_class_name,
                                                          environment=save_info.environment,
                                                          compressed_parameters=param_file)

            model_list_info = ModelListInfo(store_type=ModelListStoreType.COMPRESSED_PARAMETERS,
                                            recover_info=recover_info)

            model_info_id = model_list_info.persist(self._file_pers_service, self._dict_pers_service)

            return model_info_id

    def recover_models(self, model_list_id: str, execute_checks: bool = True) -> RestoredModelListInfo:
        recovered_models = []

        with tempfile.TemporaryDirectory() as tmp_path:
            model_info = ModelListInfo.load(model_list_id, self._file_pers_service, self._dict_pers_service,
                                            tmp_path, load_recursive=True, load_files=True)

            recover_info: CompressedModelListRecoverInfo = model_info.recover_info

            # load raw data from disk and decompress
            # read binary file
            with open(recover_info.compressed_parameters.path, 'rb') as f:
                data = f.read()

                decompression_method = self.compression_info[DECOMPRESS_FUNC]
                decompression_kwargs = self.compression_info[DECOMPRESS_KWARGS]

                if decompression_method:
                    data = decompression_method(data, **decompression_kwargs)

                # position byte array to read form
                byte_pointer = 0
                # read until we have no data left
                while byte_pointer < len(data):
                    # create a copy of the example model and load new weights in it
                    model = create_object(recover_info.model_code.path, recover_info.model_class_name)
                    model_state = model.state_dict()
                    for k, _tensor in model_state.items():
                        # the number of bytes we have to extract is equivalent to:
                        # the number of values the tensor holds * number of bytes for the datatype (for one float 4 bytes)
                        shape = _tensor.shape
                        num_bytes = math.prod(shape) * self._bytes_per_value(_tensor.dtype)
                        # read the bytes for the tensor
                        byte_data = data[byte_pointer:byte_pointer + num_bytes]
                        # form tensor out of bytes, and reshape
                        np_dtype = np.dtype(torch_dtype_to_numpy_dict[_tensor.dtype])
                        recovered_tensor = to_tensor(byte_data, np_dtype)
                        recovered_tensor = torch.reshape(recovered_tensor, shape)
                        # override the recovered tensor in the state dict
                        model_state[k] = recovered_tensor

                        # update byte pointer to read form correct position
                        byte_pointer += num_bytes

                    # as soon as state_dict for one model is recovered load it back into model and append it to result
                    model.load_state_dict(model_state)
                    recovered_models.append(model)

                return RestoredModelListInfo(models=recovered_models)

    def _bytes_per_value(self, _type: torch.dtype):
        # to be more efficient we can just create/save a lookup table
        dummy_tensor = torch.tensor([1], dtype=_type)
        return len(dummy_tensor.numpy().tobytes())


def _get_weights_hash_info(add_weights_hash_info, model_save_info):
    weights_hash_info = None
    if add_weights_hash_info:
        assert model_save_info.model, "to compute a weights info hash the a model has to be given"
        weights_hash_info = WeightDictMerkleTree.from_state_dict(model_save_info.model.state_dict())
    return weights_hash_info
