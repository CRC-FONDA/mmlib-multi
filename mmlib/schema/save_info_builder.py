import torch

from mmlib.save_info import SingleModelSaveInfo, TrainSaveInfo, ProvSingleModelSaveInfo, ModelListSaveInfo, \
    ModelSaveInfo, ProvListModelSaveInfo
from mmlib.schema.environment import Environment
from mmlib.schema.restorable_object import StateDictRestorableObjectWrapper


class ModelSaveInfoBuilder:

    def __init__(self):
        super().__init__()
        self._model = None
        self._base_model = None
        self._derived_from = None
        self._model_list = None
        self._code = None
        self._prov_raw_data = None
        self._prov_datasets_paths = None
        self._env = None
        self._prov_train_kwargs = None
        self._prov_train_service_wrapper = None

        self.general_model_info_added = False
        self.prov_model_info_added = False

    def add_train_info(self, train_service_wrapper, train_kwargs):
        self._prov_train_kwargs = train_kwargs
        self._prov_train_service_wrapper = train_service_wrapper

    def add_prov_list_info(self, derived_from, environment, dataset_paths):
        self._derived_from = derived_from
        self._env = environment
        self._prov_datasets_paths = dataset_paths

    def build_prov_list_model_save_info(self) -> ProvListModelSaveInfo:
        save_info = ProvListModelSaveInfo()
        save_info.derived_from = self._derived_from
        save_info.environment = self._env
        save_info.dataset_paths = self._prov_datasets_paths
        train_info = TrainSaveInfo(
            train_service_wrapper=self._prov_train_service_wrapper,
            train_kwargs=self._prov_train_kwargs
        )
        save_info.train_info = train_info

        return save_info

    def add_model_info(self, env: Environment, model: torch.nn.Module = None, code: str = None,
                       base_model_id: str = None, model_list: [torch.nn.Module] = None, _derived_from=None):
        """
        Adds the general model information
        :param env: The environment the training was/will be performed in.
        :param model: The actual model to save as an instance of torch.nn.Module.
        :param code: (only required if base model not given or if it can not be automatically inferred) The path to the
         code of the model .
        constructor (is needed for recover process).
        :param base_model_id: The id of the base model.
        :param model_list: List of models to be saved
        """
        self._env = env
        self._model = model
        self._base_model = base_model_id
        self._derived_from = _derived_from
        self._code = code
        self._model_list = model_list
        self.general_model_info_added = True

    def add_prov_data(self, raw_data_path: str, train_kwargs: dict,
                      train_service_wrapper: StateDictRestorableObjectWrapper):
        """
        Adds information that is required to store a model using its provenance data.
        :param raw_data_path: The path to the raw data that was used as the dataset.
        :param train_kwargs: The kwargs that will be given to the train method of the train service.
        :param train_service_wrapper: The train service wrapper that wraps the train service used to train the model.
        """
        self._prov_raw_data = raw_data_path
        self._prov_train_kwargs = train_kwargs
        self._prov_train_service_wrapper = train_service_wrapper

        self.prov_model_info_added = True

    def build(self) -> ModelSaveInfo:

        if self.general_model_info_added:
            assert self._valid_baseline_save_model_info(), 'info not sufficient'
            if self.prov_model_info_added:
                assert self._valid_prov_save_model_info(), 'info not sufficient'
                return self._build_prov_save_info()
            elif self._model_list:
                return self._build_baseline_save_model_list_info()
            else:
                return self._build_baseline_save_info()

    def _build_baseline_save_info(self):
        save_info = SingleModelSaveInfo(
            model=self._model,
            base_model=self._base_model,
            model_code=self._code,
            environment=self._env
        )

        return save_info

    def _build_baseline_save_model_list_info(self):
        save_info = ModelListSaveInfo(
            models=self._model_list,
            model_code=self._code,
            environment=self._env,
            derived_from=self._derived_from
        )

        return save_info

    def _build_prov_save_info(self):
        prov_train_info = TrainSaveInfo(
            train_service_wrapper=self._prov_train_service_wrapper,
            train_kwargs=self._prov_train_kwargs)

        save_info = ProvSingleModelSaveInfo(

            model=self._model,
            base_model=self._base_model,
            model_code=self._code,
            raw_dataset=self._prov_raw_data,
            train_info=prov_train_info,
            environment=self._env)

        return save_info

    def _valid_baseline_save_model_info(self):
        return self._model or self._base_model or self._model_list

    def _valid_prov_save_model_info(self):
        return self._valid_baseline_save_model_info() and self._base_model \
               and self._prov_raw_data and self._env and self._prov_train_service_wrapper \
               and self._prov_train_kwargs and self._prov_train_kwargs
