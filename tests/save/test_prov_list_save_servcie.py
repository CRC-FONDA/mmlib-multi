import os

import torch

from mmlib.constants import CURRENT_DATA_ROOT, MMLIB_CONFIG
from mmlib.deterministic import set_deterministic
from mmlib.equal import model_equal
from mmlib.save import ProvModelListSaveService
from mmlib.save_info import ModelListSaveInfo
from mmlib.schema.restorable_object import RestorableObjectWrapper
from mmlib.schema.save_info_builder import ModelSaveInfoBuilder
from mmlib.track_env import track_current_environment
from tests.example_files.battery_data import BatteryData
from tests.example_files.battery_dataloader import BatteryDataloader
from tests.example_files.ffnn_train import FFNNTrainWrapper, FFNNTrainService
from tests.example_files.imagenet_train import OPTIMIZER, DATALOADER, DATA
from tests.example_files.mynets.ffnn import FFNN
from tests.save.test_model_list_save_servcie import TestModelListSaveService, dummy_ffnn_input

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(FILE_PATH, '../example_files/mynets/{}.py')
CONFIG = os.path.join(FILE_PATH, '../example_files/local-config.ini')
OPTIMIZER_CODE = os.path.join(FILE_PATH, '../example_files/imagenet_optimizer.py')


class TestProvListSaveService(TestModelListSaveService):

    def setUp(self) -> None:
        super().setUp()
        assert os.path.isfile(CONFIG), \
            'to run these tests define your onw config file named \'local-config\'' \
            'to do so copy the file under tests/example_files/config.ini, and place it in the same directory' \
            'rename it to local-config.ini, create an empty directory and put the path to it in the newly created' \
            'config file, to define the current_data_root'

        os.environ[MMLIB_CONFIG] = CONFIG

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = ProvModelListSaveService(file_pers_service, dict_pers_service)

    def test_save_restore_provenance_specific_model(self):
        # take three initial models
        initial_model_list = []
        for i in range(3):
            model = FFNN()
            initial_model_list.append(model)

        # save the initial models as full models
        save_info_builder = ModelSaveInfoBuilder()
        env = track_current_environment()
        save_info_builder.add_model_info(model_list=initial_model_list, env=env)
        save_info: ModelListSaveInfo = save_info_builder.build()
        model_list_id = self.save_service.save_models(save_info)

        # recover models and check if they are equal
        restored_model_list_info = self.save_service.recover_models(model_list_id)
        for recovered_model, model in zip(restored_model_list_info.models, initial_model_list):
            self.assertTrue(model_equal(model, recovered_model, dummy_ffnn_input))

        # set deterministic for debugging purposes
        set_deterministic()

        dataset_paths = [
            os.path.abspath(os.path.join(FILE_PATH, '../example_files/data/dummy_battery_data/data_set_1')),
            os.path.abspath(os.path.join(FILE_PATH, '../example_files/data/dummy_battery_data/data_set_2')),
            os.path.abspath(os.path.join(FILE_PATH, '../example_files/data/dummy_battery_data/data_set_3'))
        ]

        ffnn_ts = FFNNTrainService()
        dataset_path = dataset_paths[0]
        current_model = initial_model_list[0]
        state_dict = self._create_ts_state_dict(current_model, dataset_path)
        ffnn_ts.state_objs = state_dict

        _prov_train_service_wrapper = FFNNTrainWrapper(instance=ffnn_ts)
        train_kwargs = {'number_epochs': 2}
        info_builder = ModelSaveInfoBuilder()
        info_builder.add_train_info(
            train_service_wrapper=_prov_train_service_wrapper,
            train_kwargs=train_kwargs
        )
        info_builder.add_prov_list_info(derived_from=model_list_id, environment=env, dataset_paths=dataset_paths)

        save_info = info_builder.build_prov_list_model_save_info()

        model_id = self.save_service.save_models(save_info)

        recovered_model_info = self.save_service.recover_models(model_id, execute_checks=True)

        for i in range(len(initial_model_list)):
            ffnn_ts = FFNNTrainService()
            dataset_path = dataset_paths[i]
            current_model = initial_model_list[i]
            state_dict = self._create_ts_state_dict(current_model, dataset_path)
            ffnn_ts.state_objs = state_dict

            ffnn_ts.train(initial_model_list[i], **train_kwargs)
            trained_model = initial_model_list[i]
            recovered_model = recovered_model_info.models[i]
            self.assertTrue(model_equal(trained_model, recovered_model, dummy_ffnn_input))

    def _create_ts_state_dict(self, current_model, dataset_path):
        state_dict = {}
        # just start with the first data root
        data_wrapper = BatteryData(root=dataset_path)
        state_dict[DATA] = RestorableObjectWrapper(
            config_args={'root': CURRENT_DATA_ROOT},
            instance=data_wrapper
        )
        data_loader_kwargs = {'batch_size': 16, 'num_workers': 0, 'pin_memory': True}
        dataloader = BatteryDataloader(data_wrapper, **data_loader_kwargs)
        state_dict[DATALOADER] = RestorableObjectWrapper(
            init_args=data_loader_kwargs,
            init_ref_type_args=['dataset'],
            instance=dataloader
        )
        optimizer_kwargs = {'lr': 10 ** -7}
        optimizer = torch.optim.SGD(current_model.parameters(), **optimizer_kwargs)
        state_dict[OPTIMIZER] = RestorableObjectWrapper(
            import_cmd='from torch.optim import SGD',
            init_args=optimizer_kwargs,
            init_ref_type_args=['params'],
            instance=optimizer
        )
        return state_dict
