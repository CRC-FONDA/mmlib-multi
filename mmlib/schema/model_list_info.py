from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.model_info import STORE_TYPE, RECOVER_INFO_ID, AbstractModelInfo, WEIGHTS_HASH_INFO
from mmlib.schema.recover_info import LIST_RECOVER_INFO, FullModelListRecoverInfo, \
    CompressedModelListRecoverInfo, ListWeightsUpdateRecoverInfo, ListProvenanceRecoverInfo
from mmlib.schema.store_type import ModelListStoreType
from mmlib.util.weight_dict_merkle_tree import WeightDictMerkleTree

MODEL_LIST_INFO = 'model_list_info'


class ModelListInfo(AbstractModelInfo):

    @property
    def _representation_type(self) -> str:
        return MODEL_LIST_INFO

    def __init__(self, store_type: ModelListStoreType = None, recover_info=None,
                 store_id: str = None, derived_from_id: str = None,
                 models_weights_hash_info: [WeightDictMerkleTree] = None):
        super().__init__(store_id, derived_from_id)
        self.store_type = store_type
        self.recover_info = recover_info
        self.models_weights_hash_info = models_weights_hash_info

    def _persist_class_specific_fields(self, dict_representation, file_pers_service, dict_pers_service):
        super()._persist_class_specific_fields(dict_representation, file_pers_service, dict_pers_service)
        recover_info_id = self.recover_info.persist(file_pers_service, dict_pers_service)

        # add mandatory fields
        dict_representation[STORE_TYPE] = self.store_type.value
        dict_representation[RECOVER_INFO_ID] = recover_info_id

        if self.models_weights_hash_info:
            dict_representation[WEIGHTS_HASH_INFO] = \
                [hash_info.to_python_dict() for hash_info in self.models_weights_hash_info]

    def load_all_fields(self, file_pers_service: FilePersistenceService,
                        dict_pers_service: DictPersistenceService, restore_root: str,
                        load_recursive: bool = False, load_files: bool = False):

        restored_dict = _recover_stored_dict(dict_pers_service, self.store_id)

        super()._load_super_fields(restored_dict)

        # mandatory fields
        if not self.store_type:
            self.store_type = _recover_store_type(restored_dict)

        if not self.recover_info or load_recursive:
            self.recover_info = _recover_recover_info(restored_dict, dict_pers_service, file_pers_service, restore_root,
                                                      self.store_type, load_recursive, load_files)

        # optional fields
        if not self.models_weights_hash_info:
            self.models_weights_hash_info = _recover_models_weights_hash_info(restored_dict)

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
    elif store_type == ModelListStoreType.COMPRESSED_PARAMETERS:
        if load_recursive:
            recover_info = CompressedModelListRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                               restore_root, load_recursive, load_files)
        else:
            recover_info = CompressedModelListRecoverInfo.load_placeholder(recover_info_id)
    elif store_type == ModelListStoreType.COMPRESSED_PARAMETERS_DIFF:
        if load_recursive:
            recover_info = ListWeightsUpdateRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                             restore_root, load_recursive, load_files)
        else:
            recover_info = ListWeightsUpdateRecoverInfo.load_placeholder(recover_info_id)
    elif store_type == ModelListStoreType.PROVENANCE:
        if load_recursive:
            recover_info = ListProvenanceRecoverInfo.load(recover_info_id, file_pers_service, dict_pers_service,
                                                          restore_root, load_recursive, load_files)
        else:
            recover_info = ListProvenanceRecoverInfo.load_placeholder(recover_info_id)
    else:
        assert False, 'Invalid store type'
    return recover_info


def _recover_models_weights_hash_info(restored_dict):
    if WEIGHTS_HASH_INFO in restored_dict:
        hash_info_list = restored_dict[WEIGHTS_HASH_INFO]
        return [WeightDictMerkleTree.from_python_dict(hash_info) for hash_info in hash_info_list]
