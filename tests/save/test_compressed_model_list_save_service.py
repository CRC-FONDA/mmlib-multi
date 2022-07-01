import zlib

from mmlib.save import CompressedModelListSaveService, COMPRESS_FUNC, COMPRESS_KWARGS, DECOMPRESS_FUNC, \
    DECOMPRESS_KWARGS
from tests.save.test_model_list_save_servcie import TestModelListSaveService


class TestCompressedModelListSaveServiceNoCompression(TestModelListSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        self.save_service = CompressedModelListSaveService(file_pers_service, dict_pers_service)


class TestCompressedModelListSaveService(TestModelListSaveService):

    def init_save_service(self, dict_pers_service, file_pers_service):
        compression_info = {
            COMPRESS_FUNC: zlib.compress,
            COMPRESS_KWARGS: {},
            DECOMPRESS_FUNC: zlib.decompress,
            DECOMPRESS_KWARGS: {}
        }
        self.save_service = CompressedModelListSaveService(file_pers_service, dict_pers_service, compression_info)
