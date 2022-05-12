import os
import tempfile

import torch

from mmlib.schema.file_reference import FileReference
from mmlib.schema.model_info import ModelInfo
from mmlib.schema.model_list_info import ModelListInfo
from mmlib.schema.recover_info import RECOVER_INFO, FullModelListRecoverInfo, LIST_RECOVER_INFO
from mmlib.schema.schema_obj import METADATA_SIZE
from mmlib.schema.store_type import ModelListStoreType
from mmlib.track_env import track_current_environment
from tests.example_files.mynets.resnet18 import resnet18
from tests.size.abstract_test_size import TestSize

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestModelListInfoSize(TestSize):

    def test_model_info_size(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = resnet18(pretrained=True)
            param_path1 = os.path.join(tmp_path, 'params1')
            param_path2 = os.path.join(tmp_path, 'params2')
            torch.save(model.state_dict(), param_path1)
            torch.save(model.state_dict(), param_path2)
            environment = track_current_environment()
            recover_info = FullModelListRecoverInfo(
                parameter_files=[FileReference(param_path1), FileReference(param_path2)],
                model_code=FileReference(os.path.join(FILE_PATH, '../example_files/mynets/resnet18.py')),
                environment=environment)

            model_info = ModelListInfo(
                store_type=ModelListStoreType.FULL_MODEL,
                recover_info=recover_info,
            )

            _id = model_info.persist(self.file_pers_service, self.dict_pers_service)

            place_holder = ModelListInfo.load_placeholder(_id)
            size_dict = place_holder.size_info(self.file_pers_service, self.dict_pers_service)

            self.assertEqual(len(size_dict.keys()), 2)
            self.assertTrue(size_dict[METADATA_SIZE] > 0)
            self.assertTrue(size_dict[LIST_RECOVER_INFO][METADATA_SIZE] > 0)
