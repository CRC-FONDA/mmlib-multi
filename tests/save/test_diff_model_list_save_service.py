import os
import shutil
import unittest

from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import DiffModelListSaveService
from mmlib.save_info import ModelListSaveInfo
from mmlib.schema.save_info_builder import ModelSaveInfoBuilder
from mmlib.track_env import track_current_environment
from mmlib.util.dummy_data import imagenet_input
from mmlib.util.mongo import MongoService
from tests.example_files.mynets.ffnn import FFNN
from tests.persistence.test_dict_persistence import MONGO_CONTAINER_NAME
from tests.save.test_model_list_save_servcie import dummy_ffnn_input


class TestDiffModelListSaveService(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = './filesystem-tmp'
        self.abs_tmp_path = os.path.abspath(self.tmp_path)

        self.__clean_up()

        # run mongo DB locally in docker container
        os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)

        self.mongo_service = MongoService('127.0.0.1', 'mmlib')

        os.mkdir(self.abs_tmp_path)
        self.file_pers_service = FileSystemPersistenceService(self.tmp_path)
        self.dict_pers_service = MongoDictPersistenceService()
        self.init_save_service(self.dict_pers_service, self.file_pers_service)

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = DiffModelListSaveService(file_pers_service, dict_pers_service)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_restore_derived_model_list(self):
        initial_model_list = []
        for i in range(3):
            model = FFNN()
            initial_model_list.append(model)

        updated_list = []

        # add equivalent model
        model = FFNN()
        model.load_state_dict(initial_model_list[0].state_dict())
        updated_list.append(model)

        # add a partially updated_model
        model = FFNN()
        state_dict = model.state_dict()
        prev_state_dict = initial_model_list[1].state_dict()
        keys = list(state_dict.keys())
        for i in range(int(len(keys) / 2)):
            key = keys[-(i + 1)]
            state_dict[key] = prev_state_dict[key]
        model.load_state_dict(state_dict)
        updated_list.append(model)

        # add a fully updated model
        model = FFNN()
        updated_list.append(model)

        initial_model_list_id = self._save_models(initial_model_list)

        derived_model_list_id = self._save_models(updated_list, derived_from=initial_model_list_id)

        restored_model_list_info = self.save_service.recover_models(initial_model_list_id)

        for recovered_model, model in zip(restored_model_list_info.models, initial_model_list):
            self.assertTrue(model_equal(model, recovered_model, dummy_ffnn_input))

        restored_model_list_info = self.save_service.recover_models(derived_model_list_id)
        for recovered_model, model in zip(restored_model_list_info.models, updated_list):
            self.assertTrue(model_equal(model, recovered_model, dummy_ffnn_input))

    def _save_models(self, model_list, derived_from=None):
        save_info_builder = ModelSaveInfoBuilder()
        env = track_current_environment()
        # save initial models
        save_info_builder.add_model_info(model_list=model_list, env=env, _derived_from=derived_from)
        save_info: ModelListSaveInfo = save_info_builder.build()
        model_list_id = self.save_service.save_models(save_info)
        return model_list_id
