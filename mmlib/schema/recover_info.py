import abc
import configparser
import os

from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT, VALUES
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.dataset import Dataset
from mmlib.schema.dataset_reference import DatasetReference
from mmlib.schema.environment import Environment
from mmlib.schema.file_reference import FileReference
from mmlib.schema.schema_obj import SchemaObj, METADATA_SIZE
from mmlib.schema.train_info import TrainInfo
from mmlib.util.helper import copy_all_data, clean

DATASETS = 'datasets'

UPDATE_LIST = 'update_list'

RECOVER_INFO = 'recover_info'
LIST_RECOVER_INFO = 'list_recover_info'

MODEL_CODE = 'model_code'
MODEL_CLASS_NAME = 'model_class_name'
PARAMETERS = 'parameters'
ENVIRONMENT = 'environment'


class AbstractRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):

    @property
    def _representation_type(self) -> str:
        return RECOVER_INFO


class FullModelRecoverInfo(AbstractRecoverInfo):

    def __init__(self, parameters_file: FileReference = None, model_code: FileReference = None,
                 model_class_name: str = None, environment: Environment = None, store_id: str = None):
        super().__init__(store_id)
        self.model_code = model_code
        self.model_class_name = model_class_name
        self.parameters_file = parameters_file
        self.environment = environment

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.model_class_name = restored_dict[MODEL_CLASS_NAME]

        self.model_code = _recover_model_code(file_pers_service, load_files, restore_root, restored_dict)
        self.parameters_file = _recover_parameters(file_pers_service, load_files, restore_root, restored_dict)
        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        file_pers_service.save_file(self.model_code)
        file_pers_service.save_file(self.parameters_file)
        env_id = self.environment.persist(file_pers_service, dict_pers_service)

        dict_representation[PARAMETERS] = self.parameters_file.reference_id
        dict_representation[MODEL_CODE] = self.model_code.reference_id
        dict_representation[MODEL_CLASS_NAME] = self.model_class_name
        dict_representation[ENVIRONMENT] = env_id

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[ENVIRONMENT] = self.environment.size_info(file_pers_service, dict_pers_service)

        file_pers_service.file_size(self.model_code)
        size_dict[MODEL_CODE] = self.model_code.size

        file_pers_service.file_size(self.parameters_file)
        size_dict[PARAMETERS] = self.parameters_file.size


def _recover_parameters(file_pers_service, load_files, restore_root, restored_dict):
    parameters_file_id = restored_dict[PARAMETERS]
    parameters_file = FileReference(reference_id=parameters_file_id)

    if load_files:
        file_pers_service.recover_file(parameters_file, restore_root)

    return parameters_file


def _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root, restored_dict):
    env_id = restored_dict[ENVIRONMENT]
    if load_recursive:
        env = Environment.load(env_id, file_pers_service, dict_pers_service, restore_root)
    else:
        env = Environment.load_placeholder(env_id)
    return env


UPDATE = 'update'
UPDATE_TYPE = 'update_type'


class WeightsUpdateRecoverInfo(AbstractRecoverInfo):

    def __init__(self, update: FileReference = None, update_type: str = None, store_id: str = None):
        super().__init__(store_id)
        self.update = update
        self.update_type = update_type

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.update = _restore_update(file_pers_service, load_files, restore_root, restored_dict)
        self.update_type = restored_dict[UPDATE_TYPE]

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        file_pers_service.save_file(self.update)
        dict_representation[UPDATE] = self.update.reference_id
        dict_representation[UPDATE_TYPE] = self.update_type

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        file_pers_service.file_size(self.update)
        size_dict[UPDATE] = self.update.size


def _restore_update(file_pers_service, load_files, restore_root, restored_dict):
    update_id = restored_dict[UPDATE]
    update = FileReference(reference_id=update_id)

    if load_files:
        file_pers_service.recover_file(update, restore_root)

    return update


class ListWeightsUpdateRecoverInfo(WeightsUpdateRecoverInfo):

    def __init__(self, update: FileReference = None, update_type: str = None, store_id: str = None,
                 update_list: list = None):
        super().__init__(update, update_type, store_id)
        self.update_list = update_list

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.update = _restore_update(file_pers_service, load_files, restore_root, restored_dict)
        self.update_type = restored_dict[UPDATE_TYPE]
        self.update_list = restored_dict[UPDATE_LIST]

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        dict_representation[UPDATE_LIST] = self.update_list


DATASET = 'dataset'
TRAIN_INFO = 'train_info'


class AbstractProvenanceRecoverInfo(AbstractRecoverInfo):
    def __init__(self, train_info: TrainInfo = None, environment: Environment = None, store_id: str = None):
        super().__init__(store_id)
        self.train_info = train_info
        self.environment = environment

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        env_id = self.environment.persist(file_pers_service, dict_pers_service)
        train_info_id = self.train_info.persist(file_pers_service, dict_pers_service)

        dict_representation[TRAIN_INFO] = train_info_id
        dict_representation[ENVIRONMENT] = env_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.train_info = _restore_train_info(
            dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive, load_files)

        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[ENVIRONMENT] = self.environment.size_info(file_pers_service, dict_pers_service)
        size_dict[TRAIN_INFO] = self.train_info.size_info(file_pers_service, dict_pers_service)


class ProvenanceRecoverInfo(AbstractProvenanceRecoverInfo):

    def __init__(self, dataset: Dataset = None, train_info: TrainInfo = None, environment: Environment = None,
                 store_id: str = None):
        super().__init__(train_info, environment, store_id)
        self.dataset = dataset

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        dataset_id = self.dataset.persist(file_pers_service, dict_pers_service)

        dict_representation[DATASET] = dataset_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        dataset_id = restored_dict[DATASET]
        self.dataset = _recover_data(dataset_id, dict_pers_service, file_pers_service, load_files, load_recursive,
                                     restore_root)

        self.train_info = _restore_train_info(
            dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive, load_files)

        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        super()._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)
        size_dict[DATASET] = self.dataset.size_info(file_pers_service, dict_pers_service)


def _data_dst_path():
    config_file = os.getenv(MMLIB_CONFIG)
    config = configparser.ConfigParser()
    config.read(config_file)

    return config[VALUES][CURRENT_DATA_ROOT]


def _recover_data(dataset_id, dict_pers_service, file_pers_service, load_files, load_recursive, restore_root):
    dataset = Dataset.load(dataset_id, file_pers_service, dict_pers_service, restore_root, load_recursive,
                           load_files)
    # make data available for train_info
    if load_files:
        # TODO for now we copy the data, maybe if we run into performance issues we should use move instead of copy
        data_dst_path = _data_dst_path()
        clean(data_dst_path)
        copy_all_data(dataset.raw_data.path, data_dst_path)
    return dataset


def _recover_model_code(file_pers_service, load_files, restore_root, restored_dict):
    model_code_id = restored_dict[MODEL_CODE]
    model_code = FileReference(reference_id=model_code_id)

    if load_files:
        file_pers_service.recover_file(model_code, restore_root)

    return model_code


def _restore_train_info(dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive,
                        load_files):
    train_info_id = restored_dict[TRAIN_INFO]
    if not load_recursive:
        train_info = TrainInfo.load_placeholder(train_info_id)
    else:
        train_info = TrainInfo.load(
            train_info_id, file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)
    return train_info


class AbstractListRecoverInfo(SchemaObj, metaclass=abc.ABCMeta):

    @property
    def _representation_type(self) -> str:
        return LIST_RECOVER_INFO

    def __init__(self, model_code: FileReference = None, model_class_name: str = None, environment: Environment = None,
                 store_id: str = None):
        super().__init__(store_id)
        self.model_code = model_code
        self.model_class_name = model_class_name
        self.environment = environment

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        file_pers_service.save_file(self.model_code)

        env_id = self.environment.persist(file_pers_service, dict_pers_service)

        dict_representation[MODEL_CODE] = self.model_code.reference_id
        dict_representation[MODEL_CLASS_NAME] = self.model_class_name
        dict_representation[ENVIRONMENT] = env_id

    def _load_abstract_fields(self, restored_dict, file_pers_service: FilePersistenceService,
                              dict_pers_service: DictPersistenceService, restore_root: str, load_recursive: bool = True,
                              load_files: bool = True):
        self.model_class_name = restored_dict[MODEL_CLASS_NAME]

        self.model_code = _recover_model_code(file_pers_service, load_files, restore_root, restored_dict)
        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[ENVIRONMENT] = self.environment.size_info(file_pers_service, dict_pers_service)

        file_pers_service.file_size(self.model_code)
        size_dict[MODEL_CODE] = self.model_code.size


class FullModelListRecoverInfo(AbstractListRecoverInfo):

    def __init__(self, parameter_files: [FileReference] = None, model_code: FileReference = None,
                 model_class_name: str = None, environment: Environment = None, store_id: str = None):
        super().__init__(model_code, model_class_name, environment, store_id)
        self.parameter_files = parameter_files

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, self._representation_type)

        super()._load_abstract_fields(restored_dict, file_pers_service, dict_pers_service, restore_root, load_recursive,
                                      load_files)

        self.parameter_files = _recover_parameter_list(file_pers_service, load_files, restore_root, restored_dict)

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        for parameter_file in self.parameter_files:
            file_pers_service.save_file(parameter_file)

        dict_representation[PARAMETERS] = [pf.reference_id for pf in self.parameter_files]

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        super()._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)

        param_sizes = []
        for parameter_file in self.parameter_files:
            file_pers_service.file_size(parameter_file)
            param_sizes.append(parameter_file.size)

        size_dict[PARAMETERS] = sum(param_sizes)


def _recover_parameter_list(file_pers_service, load_files, restore_root, restored_dict):
    parameter_files = []
    for parameters_file_id in restored_dict[PARAMETERS]:
        parameters_file = FileReference(reference_id=parameters_file_id)

        if load_files:
            file_pers_service.recover_file(parameters_file, restore_root)

        parameter_files.append(parameters_file)

    return parameter_files


class CompressedModelListRecoverInfo(AbstractListRecoverInfo):

    def __init__(self, compressed_parameters: FileReference = None, model_code: FileReference = None,
                 model_class_name: str = None, environment: Environment = None, store_id: str = None):
        super().__init__(model_code, model_class_name, environment, store_id)
        self.compressed_parameters = compressed_parameters

    def load_all_fields(self, file_pers_service: FilePersistenceService, dict_pers_service: DictPersistenceService,
                        restore_root: str, load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, self._representation_type)

        super()._load_abstract_fields(restored_dict, file_pers_service, dict_pers_service, restore_root, load_recursive,
                                      load_files)

        self.compressed_parameters = _recover_compressed_parameters(file_pers_service, load_files, restore_root,
                                                                    restored_dict)

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)

        file_pers_service.save_file(self.compressed_parameters)
        dict_representation[PARAMETERS] = self.compressed_parameters.reference_id

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        super()._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)

        file_pers_service.file_size(self.compressed_parameters)
        size_dict[PARAMETERS] = self.compressed_parameters.size


class ListProvenanceRecoverInfo(AbstractProvenanceRecoverInfo):

    def __init__(self, datasets: [DatasetReference] = None, train_info: TrainInfo = None, environment: Environment = None,
                 store_id: str = None):
        super().__init__(train_info, environment, store_id)
        self.datasets = datasets

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        dataset_ids = []
        for dataset in self.datasets:
            dataset_id = dataset.persist(file_pers_service, dict_pers_service)
            dataset_ids.append(dataset_id)

        dict_representation[DATASETS] = dataset_ids

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        self.datasets = \
            [DatasetReference.load(_id, file_pers_service, dict_pers_service, restore_root, load_recursive, load_files)
             for _id in restored_dict[DATASETS]]
        self._load_for_dataset_index(dict_pers_service, file_pers_service, load_files, load_recursive,
                                     restore_root, 0, restored_dict)

    def _load_for_dataset_index(self, dict_pers_service, file_pers_service, load_files, load_recursive,
                                restore_root, dataset_index, restored_dict=None):

        if restored_dict is None:
            restored_dict = dict_pers_service.recover_dict(self.store_id, RECOVER_INFO)

        _copy_data_to_data_root(self.datasets[dataset_index])

        self.train_info = _restore_train_info(
            dict_pers_service, file_pers_service, restore_root, restored_dict, load_recursive, load_files)

        self.environment = _recover_environment(dict_pers_service, file_pers_service, load_recursive, restore_root,
                                                restored_dict)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        super()._add_reference_sizes(size_dict, file_pers_service, dict_pers_service)
        size_sum = 0
        for dataset in self.datasets:
            dataset_size_info = dataset.size_info(file_pers_service, dict_pers_service)
            # WARNING here we only consider the size of the metadat and references, not of the underlying data
            size_sum += dataset_size_info[METADATA_SIZE]

        size_dict[DATASETS] = size_sum

    def adjust_for_dataset(self, dict_pers_service, file_pers_service, load_files, load_recursive,
                           restore_root, dataset_index):
        self._load_for_dataset_index(dict_pers_service, file_pers_service, load_files, load_recursive,
                                     restore_root, dataset_index, restored_dict=None)


def _copy_data_to_data_root(dataset_ref: DatasetReference):
    data_dst_path = _data_dst_path()
    clean(data_dst_path)
    copy_all_data(dataset_ref.data_path, data_dst_path)


def _recover_compressed_parameters(file_pers_service, load_files, restore_root, restored_dict):
    update_id = restored_dict[PARAMETERS]
    update = FileReference(reference_id=update_id)

    if load_files:
        file_pers_service.recover_file(update, restore_root)

    return update
