import unittest

from mmlib.save import CompressedModelListSaveService
from tests.save.test_model_list_save_servcie import TestModelListSaveService


class TestCompressedModelListSaveService(TestModelListSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = CompressedModelListSaveService(file_pers_service, dict_pers_service, logging=True)
