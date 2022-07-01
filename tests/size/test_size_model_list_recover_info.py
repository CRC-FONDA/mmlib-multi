import os
import shutil

import torch

from mmlib.schema.file_reference import FileReference
from mmlib.schema.recover_info import FullModelListRecoverInfo, CompressedModelListRecoverInfo, \
    ENVIRONMENT, MODEL_CODE, PARAMETERS
from mmlib.schema.schema_obj import METADATA_SIZE
from mmlib.track_env import track_current_environment
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

TMP_DIR = './tmp-dir'

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestModelListInfoSize(TestSize):

    def setUp(self) -> None:
        super().setUp()
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        self.param_path1 = os.path.join(TMP_DIR, 'params1')
        self.param_path2 = os.path.join(TMP_DIR, 'params2')
        model = resnet18(pretrained=True)
        torch.save(model.state_dict(), self.param_path1)
        model = resnet18()
        torch.save(model.state_dict(), self.param_path2)
        self.environment = track_current_environment()

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(TMP_DIR)

    def test_full_model_recover_info_size(self):
        recover_info = FullModelListRecoverInfo(
            parameter_files=[FileReference(self.param_path1), FileReference(self.param_path2)],
            model_code=FileReference(os.path.join(FILE_PATH, '../example_files/mynets/resnet18.py')),
            environment=self.environment)

        _id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = FullModelListRecoverInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        self.assertEqual(4, len(size_dict.keys()))
        self.assertTrue(size_dict[ENVIRONMENT][METADATA_SIZE] > 0)
        self.assertTrue(size_dict[MODEL_CODE] > 0)
        self.assertEqual(2 * os.path.getsize(self.param_path1), size_dict[PARAMETERS])

    def test_compressed_model_recover_info_size(self):
        recover_info = CompressedModelListRecoverInfo(
            compressed_parameters=FileReference(self.param_path1),
            model_code=FileReference(os.path.join(FILE_PATH, '../example_files/mynets/resnet18.py')),
            environment=self.environment)

        _id = recover_info.persist(self.file_pers_service, self.dict_pers_service)

        place_holder = CompressedModelListRecoverInfo.load_placeholder(_id)
        size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

        self.assertEqual(4, len(size_dict.keys()))
        self.assertTrue(size_dict[ENVIRONMENT][METADATA_SIZE] > 0)
        self.assertTrue(size_dict[MODEL_CODE] > 0)
        self.assertEqual(os.path.getsize(self.param_path1), size_dict[PARAMETERS])
