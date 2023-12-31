from enum import Enum


class ModelStoreType(Enum):
    FULL_MODEL = '1'
    WEIGHT_UPDATES = '2'
    PROVENANCE = '3'


class ModelListStoreType(Enum):
    FULL_MODEL = '1'
    COMPRESSED_PARAMETERS = '2'
    COMPRESSED_PARAMETERS_DIFF = '3'
    PROVENANCE = '4'
