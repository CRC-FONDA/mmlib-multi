import os
import shutil
import unittest

import torch

from mmlib.equal import model_equal
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ModelListSaveService
from mmlib.save_info import ModelListSaveInfo
from mmlib.schema.save_info_builder import ModelSaveInfoBuilder
from mmlib.track_env import track_current_environment
from mmlib.util.dummy_data import imagenet_input
from mmlib.util.mongo import MongoService
from tests.example_files.mynets.mobilenet import mobilenet_v2
from tests.example_files.mynets.resnet18 import resnet18
from tests.persistence.test_dict_persistence import MONGO_CONTAINER_NAME


def dummy_ffnn_input(batch_size: int = 10) -> torch.tensor:
    """
    Generates a batch of dummy imputes for models processing imagenet data.
    :param batch_size: The size of the batch.
    :return: Returns a tensor containing the generated batch.
    """
    batch = []
    for i in range(batch_size):
        batch.append(torch.rand(5))
    return torch.stack(batch)


class TestModelListSaveService(unittest.TestCase):

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
        self.save_service = ModelListSaveService(file_pers_service, dict_pers_service, logging=True)

    def tearDown(self) -> None:
        self.__clean_up()

    def __clean_up(self):
        os.system('docker kill %s' % MONGO_CONTAINER_NAME)
        if os.path.exists(self.abs_tmp_path):
            shutil.rmtree(self.abs_tmp_path)

    def test_save_restore_resnet18_list(self):
        model_list = []
        for i in range(3):
            model = resnet18()
            model_list.append(model)

        self._test_save_restore_model_list(model_list)

    def test_save_restore_mobilenet_list(self):
        model_list = []
        for i in range(3):
            model = mobilenet_v2()
            model_list.append(model)

        self._test_save_restore_model_list(model_list)

    def _test_save_restore_model_list(self, model_list):
        save_info_builder = ModelSaveInfoBuilder()
        env = track_current_environment()
        save_info_builder.add_model_info(model_list=model_list, env=env)
        save_info: ModelListSaveInfo = save_info_builder.build()
        model_list_id = self.save_service.save_models(save_info)
        restored_model_list_info = self.save_service.recover_models(model_list_id)
        for recovered_model, model in zip(restored_model_list_info.models, model_list):
            self.assertTrue(model_equal(model, recovered_model, imagenet_input))
