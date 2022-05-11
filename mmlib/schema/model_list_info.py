from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.model_info import STORE_TYPE, RECOVER_INFO_ID
from mmlib.schema.recover_info import AbstractListRecoverInfo, LIST_RECOVER_INFO, FullModelListRecoverInfo
from mmlib.schema.schema_obj import SchemaObj
from mmlib.schema.store_type import ModelListStoreType

MODEL_LIST_INFO = 'model_list_info'


class ModelListInfo(SchemaObj):

    @property
    def _representation_type(self) -> str:
        return MODEL_LIST_INFO

    def __init__(self, store_type: ModelListStoreType = None, recover_info: AbstractListRecoverInfo = None,
                 store_id: str = None):
        super().__init__(store_id)
        self.store_type = store_type
        self.recover_info = recover_info

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        recover_info_id = self.recover_info.persist(file_pers_service, dict_pers_service)

        # add mandatory fields
        dict_representation[STORE_TYPE] = self.store_type.value
        dict_representation[RECOVER_INFO_ID] = recover_info_id

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = False, load_files: bool = False):

        restored_dict = _recover_stored_dict(dict_pers_service, self.store_id)

        # mandatory fields
        if not self.store_type:
            self.store_type = _recover_store_type(restored_dict)

        if not self.recover_info or load_recursive:
            self.recover_info = _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root,
                                                      self.store_type, load_recursive, load_files)

    def _add_reference_sizes(self, size_dict, file_pers_service, dict_pers_service):
        size_dict[LIST_RECOVER_INFO] = self.recover_info.size_info(file_pers_service, dict_pers_service)


def _recover_store_type(restored_dict):
    return ModelListStoreType(restored_dict[STORE_TYPE])


def _recover_stored_dict(dict_pers_service, obj_id):
    return dict_pers_service.recover_dict(obj_id, MODEL_LIST_INFO)


def _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root, store_type, load_recursive,
                          load_files):
    recover_info_id = restored_dict[RECOVER_INFO_ID]

    if store_type == ModelListStoreType.FULL_MODEL:
        if load_recursive:
            recover_info = FullModelListRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                         restore_root, load_recursive, load_files)
        else:
            recover_info = FullModelListRecoverInfo.load_placeholder(recover_info_id)
    else:
        assert False, 'Invalid store type'
    return recover_info
