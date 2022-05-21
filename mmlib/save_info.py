import torch

from mmlib.schema.environment import Environment
from mmlib.schema.restorable_object import StateDictRestorableObjectWrapper
from mmlib.util.helper import class_name, source_file


class TrainSaveInfo:
    def __init__(self, train_service_wrapper: StateDictRestorableObjectWrapper, train_kwargs: dict):
        self.train_service = train_service_wrapper.instance
        self.train_wrapper_code = source_file(train_service_wrapper)
        self.train_wrapper_class_name = class_name(train_service_wrapper)
        self.train_kwargs = train_kwargs


class ModelSaveInfo:
    def __init__(self, environment: Environment):
        self.environment = environment


class SingleModelSaveInfo(ModelSaveInfo):
    def __init__(self, model: torch.nn.Module, base_model: str, environment: Environment, model_code: str = None):
        super().__init__(environment)
        self.model = model
        self.base_model = base_model
        if model:
            self.model_class_name = class_name(model)
            self.model_code = model_code if model_code else source_file(model)


class ModelListSaveInfo(ModelSaveInfo):
    def __init__(self, models: [torch.nn.Module], environment: Environment, model_code: str = None,
                 derived_from: str = None):
        super().__init__(environment)
        self.models = models
        self.derived_from = derived_from
        if models[0]:
            self.model_class_name = class_name(models[0])
            self.model_code = self._get_model_code(model_code, models[0])
            if len(models) > 1:
                assert self.model_class_name == class_name(models[1])
                assert self.model_code == self._get_model_code(model_code, models[1])

    def _get_model_code(self, model_code, model):
        return model_code if model_code else source_file(model)


class ProvSingleModelSaveInfo(SingleModelSaveInfo):
    def __init__(self, model: torch.nn.Module, base_model: str, model_code: str, raw_dataset: str,
                 train_info: TrainSaveInfo, environment: Environment):
        super().__init__(model, base_model, environment, model_code)
        self.raw_dataset = raw_dataset
        self.train_info = train_info


class ProvListModelSaveInfo(ModelListSaveInfo):
    def __init__(self, models: [torch.nn.Module] = None, derived_from: str = None, model_code: str = None,
                 dataset_paths: [str] = None, train_info: TrainSaveInfo = None, environment: Environment = None):
        super().__init__(models, environment, model_code, derived_from)
        self.dataset_paths = dataset_paths
        self.train_info = train_info
