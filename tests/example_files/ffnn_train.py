import torch

from mmlib.deterministic import set_deterministic
from mmlib.persistence import FilePersistenceService, DictPersistenceService
from mmlib.schema.restorable_object import TrainService, StateDictRestorableObjectWrapper, \
    RestorableObjectWrapper, StateFileRestorableObjectWrapper
from mmlib.util.init_from_file import create_object

DATA = 'data'
DATALOADER = 'dataloader'
OPTIMIZER = 'optimizer'


class FFNNTrainService(TrainService):
    def train(self, model: torch.nn.Module, number_epochs=None):

        set_deterministic()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dataloader = self._get_dataloader()
        optimizer = self._get_optimizer(model.parameters())

        model.to(device)

        # switch to train mode
        model.train()

        for epoch in range(number_epochs):
            # iterate over data
            for idx, (_input, _label) in enumerate(dataloader):
                # reset the gradient in the optimizer
                optimizer.zero_grad()
                # let the model make a prediction in the given data
                predicted_label = model(_input)
                # calculate the loss using the ground truth
                loss = torch.nn.MSELoss(predicted_label, _label)
                # backpropagation
                loss.backward()
                # adjust the model parameters
                optimizer.step()

    def _get_dataloader(self):
        dataloader = self.state_objs[DATALOADER].instance
        return dataloader

    def _get_optimizer(self, parameters):
        optimizer_wrapper: StateFileRestorableObjectWrapper = self.state_objs[OPTIMIZER]

        if not optimizer_wrapper.instance:
            optimizer_wrapper.restore_instance({'params': parameters})

        return optimizer_wrapper.instance


class FFNNTrainWrapper(StateDictRestorableObjectWrapper):

    def restore_instance(self, file_pers_service: FilePersistenceService,
                         dict_pers_service: DictPersistenceService, restore_root: str, ):
        state_dict = {}

        optimizer = RestorableObjectWrapper.load(
            self.state_objs[OPTIMIZER], file_pers_service, dict_pers_service, restore_root, True, True)
        state_dict[OPTIMIZER] = optimizer
        optimizer.restore_instance()

        data_wrapper = RestorableObjectWrapper.load(
            self.state_objs[DATA], file_pers_service, dict_pers_service, restore_root, True, True)
        state_dict[DATA] = data_wrapper
        data_wrapper.restore_instance()

        # NOTE: Dataloader instance is loaded in the train routine
        dataloader = RestorableObjectWrapper.load(
            self.state_objs[DATALOADER], file_pers_service, dict_pers_service, restore_root, True, True)
        state_dict[DATALOADER] = dataloader
        dataloader.restore_instance(ref_type_args={'dataset': data_wrapper.instance})

        self.instance = create_object(code=self.code.path, class_name=self.class_name)
        self.instance.state_objs = state_dict
