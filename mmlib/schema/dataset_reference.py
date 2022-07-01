from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.schema_obj import SchemaObj

DATASET_REFERENCE = 'DATASET_REFERENCE'

DATA_PATH = 'data_path'


class DatasetReference(SchemaObj):

    def __init__(self, data_path: str = None, store_id: str = None):
        super().__init__(store_id)
        self.data_path = data_path

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        dict_representation[DATA_PATH] = self.data_path

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = True, load_files: bool = True):
        restored_dict = dict_pers_service.recover_dict(self.store_id, DATASET_REFERENCE)

        self.data_path = restored_dict[DATA_PATH]

    @property
    def _representation_type(self) -> str:
        return DATASET_REFERENCE

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        pass
