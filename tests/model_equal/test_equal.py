import unittest

import torch
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.equal import state_dict_equal, model_equal
from mmlib.util.dummy_data import imagenet_input
from mmlib.util.hash import state_dict_hash, tensor_hash


class TestStateDictEqual(unittest.TestCase):

    def test_empty_dicts(self):
        d1 = {}
        d2 = {}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_same_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d1))

    def test_equal_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}
        d2 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_different_keys(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test1': tensor}
        d2 = {'test2': tensor}

        self.assertFalse(state_dict_equal(d1, d2))

    def test_different_tensor(self):
        tensor1 = torch.rand(3, 300, 400)
        tensor2 = torch.rand(3, 300, 400)

        d1 = {'test': tensor1}
        d2 = {'test': tensor2}

        self.assertFalse(state_dict_equal(d1, d2))


class TestModelEqual(unittest.TestCase):

    def test_resnet18_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.resnet18(pretrained=True)

        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_googlenet_pretrained(self):
        mod1 = models.googlenet(pretrained=True)
        mod2 = models.googlenet(pretrained=True)

        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_mobilenet_v2_pretrained(self):
        mod1 = models.mobilenet_v2(pretrained=True)
        mod2 = models.mobilenet_v2(pretrained=True)

        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_resnet18_mobilenet_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.mobilenet_v2(pretrained=True)

        self.assertFalse(model_equal(mod1, mod2, imagenet_input))

    def test_not_pretrained(self):
        mod1 = models.resnet18()
        mod2 = models.resnet18()

        # we expect this to be false since the weight initialization is random
        self.assertFalse(model_equal(mod1, mod2, imagenet_input))

    def test_resnet18_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.resnet18()

        set_deterministic()
        mod2 = models.resnet18()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_googlenet_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.googlenet()

        set_deterministic()
        mod2 = models.googlenet()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_mobilenet_v2_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.mobilenet_v2()

        set_deterministic()
        mod2 = models.mobilenet_v2()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(model_equal(mod1, mod2, imagenet_input))

    def test_not_pretrained_deterministic_multiple_models(self):
        set_deterministic()
        alex1 = models.alexnet()
        resnet1 = models.resnet18()

        set_deterministic()
        alex2 = models.alexnet()
        resnet2 = models.resnet18()

        self.assertTrue(model_equal(alex1, alex2, imagenet_input))
        self.assertTrue(model_equal(resnet1, resnet2, imagenet_input))

    def test_resnet18_state_dict_hash(self):
        set_deterministic()
        mod1 = models.resnet18()
        set_deterministic()
        mod2 = models.resnet18()

        mod1_dict = mod1.state_dict()
        hash1 = state_dict_hash(mod1_dict)

        mod2_dict = mod2.state_dict()
        hash2 = state_dict_hash(mod2_dict)

        # because of deterministic weight initialization we should get the same weight dicts and thus the same hashes
        self.assertEqual(hash1, hash2)

    def test_mobilenet_state_dict_hash(self):
        set_deterministic()
        mod1 = models.mobilenet_v2()
        mod1_dict = mod1.state_dict()
        hash1 = state_dict_hash(mod1_dict)

        set_deterministic()
        mod2 = models.mobilenet_v2()
        mod2_dict = mod2.state_dict()
        hash2 = state_dict_hash(mod2_dict)

        # because of deterministic weight initialization we should get the same weight dicts and thus the same hashes
        self.assertEqual(hash1, hash2)

    def test_mobilenet_state_dict_hash_diff(self):
        mod1 = models.mobilenet_v2()
        mod1_dict = mod1.state_dict()
        hash1 = state_dict_hash(mod1_dict)

        mod2 = models.mobilenet_v2()
        mod2_dict = mod2.state_dict()
        hash2 = state_dict_hash(mod2_dict)

        # we expect different results, because wight initialization is random
        # and we don't ensure deterministic execution
        self.assertNotEqual(hash1, hash2)

    def test_googlenet_state_dict_hash(self):
        set_deterministic()
        mod1 = models.googlenet()
        mod1_dict = mod1.state_dict()
        hash1 = state_dict_hash(mod1_dict)

        set_deterministic()
        mod2 = models.googlenet()
        mod2_dict = mod2.state_dict()
        hash2 = state_dict_hash(mod2_dict)

        # because of deterministic weight initialization we should get the same weight dicts and thus the same hashes
        self.assertEqual(hash1, hash2)

    def test_hash_tensor(self):
        set_deterministic()
        t1 = torch.rand(100, 100, 100)
        set_deterministic()
        t2 = torch.rand(100, 100, 100)
        t3 = torch.rand(100, 100, 100)

        t1_hash = tensor_hash(t1)
        t2_hash = tensor_hash(t2)
        t3_hash = tensor_hash(t3)

        self.assertEqual(t1_hash, t2_hash)
        self.assertNotEqual(t1_hash, t3_hash)
        self.assertNotEqual(t2_hash, t3_hash)
