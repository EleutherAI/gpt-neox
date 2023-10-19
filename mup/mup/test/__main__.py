import itertools
import unittest
from functools import partial
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mup.coord_check import get_coord_data
from mup.optim import MuAdam, MuSGD
from mup.shape import get_infshapes, get_shapes, make_base_shapes, set_base_shapes
from mup.test.models import (generate_CNN, generate_MLP, _generate_MLP, get_lazy_models,
                             get_train_loader, init_methods)

train_loader = get_train_loader(batch_size=32, num_workers=4, download=True)

def reset_seed():
    torch.manual_seed(0)

class SetBaseShapeCase(unittest.TestCase):
    mlp_base_shapes_file = 'mlp64.bsh.test'

    def get_mlp_infshapes1(self):
        base_model = _generate_MLP(64, True, True, True)
        delta_model = _generate_MLP(65, True, True, True)
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, base_model, delta=delta_model, savefile=self.mlp_base_shapes_file)
        return get_infshapes(target_model)
    
    def get_mlp_infshapes1meta(self):
        base_model = _generate_MLP(64, True, True, True, device='meta')
        delta_model = _generate_MLP(65, True, True, True, device='meta')
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, base_model, delta=delta_model, savefile=self.mlp_base_shapes_file)
        return get_infshapes(target_model)

    def get_mlp_infshapes2(self):
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, self.mlp_base_shapes_file)
        return get_infshapes(target_model)

    def get_mlp_infshapes3(self):
        base_model = _generate_MLP(64, True, True, True)
        delta_model = _generate_MLP(65, True, True, True)
        base_infshapes = make_base_shapes(base_model, delta_model)
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, base_infshapes)
        return get_infshapes(target_model)
    
    def get_mlp_infshapes3meta(self):
        base_model = _generate_MLP(64, True, True, True, device='meta')
        delta_model = _generate_MLP(65, True, True, True, device='meta')
        base_infshapes = make_base_shapes(base_model, delta_model)
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, base_infshapes)
        return get_infshapes(target_model)

    def get_mlp_infshapes4(self):
        base_model = _generate_MLP(64, True, True, True)
        delta_model = _generate_MLP(65, True, True, True)
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, get_shapes(base_model), delta=get_shapes(delta_model))
        return get_infshapes(target_model)
        
    def get_mlp_infshapes4meta(self):
        base_model = _generate_MLP(64, True, True, True)
        delta_model = _generate_MLP(65, True, True, True, device='meta')
        target_model = _generate_MLP(128, True, True, True, device='meta')
        set_base_shapes(target_model, get_shapes(base_model), delta=get_shapes(delta_model))
        return get_infshapes(target_model)

    def get_mlp_infshapes5(self):
        delta_model = _generate_MLP(65, True, True, True)
        target_model = _generate_MLP(128, True, True, True)
        # `delta` here doesn't do anything because of base shape file
        set_base_shapes(target_model, self.mlp_base_shapes_file, delta=get_shapes(delta_model))
        return get_infshapes(target_model)

    def get_mlp_infshapes5meta(self):
        delta_model = _generate_MLP(65, True, True, True, device='meta')
        target_model = _generate_MLP(128, True, True, True)
        # `delta` here doesn't do anything because of base shape file
        set_base_shapes(target_model, self.mlp_base_shapes_file, delta=get_shapes(delta_model))
        return get_infshapes(target_model)

    def get_mlp_infshapes_bad(self):
        base_model = _generate_MLP(64, True, True, True)
        target_model = _generate_MLP(128, True, True, True)
        set_base_shapes(target_model, base_model, delta=base_model)
        return get_infshapes(target_model)

    def test_set_base_shape(self):
        self.assertEqual(self.get_mlp_infshapes1(), self.get_mlp_infshapes1meta())
        self.assertEqual(self.get_mlp_infshapes1(), self.get_mlp_infshapes2())
        self.assertEqual(self.get_mlp_infshapes3(), self.get_mlp_infshapes2())
        self.assertEqual(self.get_mlp_infshapes3(), self.get_mlp_infshapes4())
        self.assertEqual(self.get_mlp_infshapes3(), self.get_mlp_infshapes3meta())
        self.assertEqual(self.get_mlp_infshapes4(), self.get_mlp_infshapes4meta())
        self.assertEqual(self.get_mlp_infshapes5(), self.get_mlp_infshapes4())
        self.assertEqual(self.get_mlp_infshapes5(), self.get_mlp_infshapes5meta())
        self.assertNotEqual(self.get_mlp_infshapes5(), self.get_mlp_infshapes_bad())


class BackwardCompatibleCase(unittest.TestCase):

    def gen_model(self, arch, width, batchnorm=False, mup=True):
        if arch == 'mlp':
            return generate_MLP(width=width, batchnorm=batchnorm, readout_zero_init=False, base_width=256, mup=mup)
        elif arch == 'cnn':
            return generate_CNN(width=width, batchnorm=batchnorm, readout_zero_init=False, base_width=8, mup=mup)
        else:
            raise ValueError()

    def test_MLP_CNN_at_base_width(self):
        for arch, batchnorm in itertools.product(['mlp', 'cnn'], [False, True]):
            for init_name, init in init_methods.items():
                reset_seed()
                mup_model = self.gen_model('mlp', 256, mup=True, batchnorm=batchnorm)
                reset_seed()
                init(mup_model)
                reset_seed()
                SP_model = self.gen_model('mlp', 256, mup=False, batchnorm=batchnorm)
                reset_seed()
                init(SP_model)
                for (name, mup_param), (_, SP_param) in zip(
                        mup_model.named_parameters(), SP_model.named_parameters()):
                    with self.subTest(name=f'{arch}, {name}, {init_name}, bn={batchnorm}'):
                        self.assertEqual((mup_param.data - SP_param.data).abs().sum().item(), 0)
    
    def test_MLP_at_diff_width_init(self):
        for init_name, init in init_methods.items():
            reset_seed()
            mup_model = self.gen_model('mlp', 128, mup=True)
            reset_seed()
            init(mup_model)
            reset_seed()
            SP_model = self.gen_model('mlp', 128, mup=False)
            reset_seed()
            init(SP_model)
            
            mup_params = dict(mup_model.named_parameters())
            SP_params = dict(SP_model.named_parameters())
            
            if init_name == 'default' or 'fan_in' in init_name:
                diff_names = ['2.bias', '4.bias', '4.weight']
                same_names = ['0.weight', '0.bias', '2.weight']
            elif 'fan_out' in init_name:
                diff_names = ['2.bias', '4.bias', '0.weight']
                same_names = ['4.weight', '0.bias', '2.weight']
            elif 'xavier' in init_name:
                diff_names = ['2.bias', '4.bias', '0.weight', '4.weight']
                same_names = ['0.bias', '2.weight']
            elif 'const' in init_name:
                diff_names = ['2.bias', '4.bias', '2.weight']
                same_names = ['0.weight', '0.bias', '4.weight']
            else:
                raise ValueError()

            for name in diff_names:
                with self.subTest(name=f'{name}, {init_name}'):
                    self.assertNotEqual(
                        (mup_params[name] - SP_params[name]).abs().sum().item(), 0)
            for name in same_names:
                with self.subTest(name=f'{name}, {init_name}'):
                    self.assertEqual(
                        (mup_params[name] - SP_params[name]).abs().sum().item(), 0)

    def test_CNN_at_diff_width_init(self):
        for init_name, init in init_methods.items():
            reset_seed()
            mup_model = self.gen_model('cnn', 16, mup=True)
            reset_seed()
            init(mup_model)
            reset_seed()
            SP_model = self.gen_model('cnn', 16, mup=False)
            reset_seed()
            init(SP_model)
            
            mup_params = dict(mup_model.named_parameters())
            SP_params = dict(SP_model.named_parameters())
                 
            if init_name == 'default' or 'fan_in' in init_name:
                diff_names = ['3.bias', '7.bias', '9.bias', '11.bias', '11.weight']
                same_names = ['0.bias', '0.weight', '3.weight', '7.weight', '9.weight']
            elif 'fan_out' in init_name:
                diff_names = ['3.bias', '7.bias', '9.bias', '11.bias', '0.weight']
                same_names = ['0.bias', '3.weight', '7.weight', '9.weight', '11.weight']
            elif 'xavier' in init_name:
                diff_names = ['3.bias', '7.bias', '9.bias', '11.bias', '0.weight', '11.weight']
                same_names = ['0.bias', '3.weight', '7.weight', '9.weight']
            elif 'const' in init_name:
                diff_names = ['3.bias', '7.bias', '9.bias', '11.bias', '3.weight', '7.weight', '9.weight']
                same_names = ['0.bias', '0.weight', '11.weight']
            else:
                raise ValueError()

            for name in diff_names:
                with self.subTest(name=f'{name}, {init_name}'):
                    self.assertNotEqual(
                        (mup_params[name] - SP_params[name]).abs().sum().item(), 0)
            for name in same_names:
                with self.subTest(name=f'{name}, {init_name}'):
                    self.assertEqual(
                        (mup_params[name] - SP_params[name]).abs().sum().item(), 0)

def train_model(model, train_loader, step=-1, optcls=MuSGD, lr=0.1, flatten_input=False, cuda=True):
    model.train()
    train_loss = 0
    train_losses = []
    optimizer = optcls(model.parameters(), lr=lr)
    for batch_idx, (data, target) in enumerate(cycle(iter(train_loader)), 1):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if flatten_input:
            data = data.view(data.size(0), -1)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        train_loss += loss.item()
        train_losses.append(train_loss / batch_idx)
        optimizer.step()
        if batch_idx == step: break
    # train_loss /= batch_idx
    return train_losses

train_model_MuSGD = partial(train_model, optcls=MuSGD, lr=0.1)
train_model_MuAdam = partial(train_model, optcls=MuAdam, lr=1e-3)

class CoordCheckCase(unittest.TestCase):

    def test_MLP_CNN(self):
        combos = list(itertools.product(['mlp', 'cnn'], [True], [False, True], ['sgd', 'adam'], init_methods.keys()))
        # comment out the following 2 lines to do all tests
        idx = np.random.choice(np.arange(len(combos)), size=10)
        combos = np.array(combos)[idx]
        for arch, mup, batchnorm, optimizer, init in combos:
            widths = [128, 512] if arch == 'cnn' else [1000, 4000]
            models = get_lazy_models(arch, widths, mup=mup, batchnorm=batchnorm, init=init)
            df = get_coord_data(models, train_loader, mup=mup, optimizer=optimizer, flatten_input=arch == 'mlp')
            df = df[df.module != '']
            df['module'] = pd.to_numeric(df['module'])
            for t, module in itertools.product([1, 2, 3], df['module'].unique()):
                with self.subTest(
                        name=f'{arch}, mup={mup}, bn={batchnorm}, {optimizer}, {init}, t={t}, module={module}'):
                    data = df[(df['module'] == module) & (df['t'] == t)]
                    std0 = data[data.width==widths[0]]['l1'].unique()[0]
                    std1 = data[data.width==widths[1]]['l1'].unique()[0]
                    if t == 1 and module == df['module'].max():
                        self.assertTrue(std0 == std1 == 0,
                        f'output should be 0 due to readout_zero_init: {std0}, {std1}')
                    else:
                        tol = 1.2
                        self.assertGreater(std1/std0, 1/tol, f'{std0}, {std1}')
                        self.assertLess(std1/std0, tol, f'{std0}, {std1}')


class MLPTrainCase(unittest.TestCase):

    def train_adam(self, model, step):
        return train_model_MuAdam(model, train_loader, step=step, flatten_input=True)

    def train_sgd(self, model, step):
        return train_model_MuSGD(model, train_loader, step=step, flatten_input=True)

    def setUp(self):
        self.models = {w: generate_MLP(w, bias=True, readout_zero_init=True, base_width=256, init='kaiming_fan_in_normal', bias_zero_init=True).cuda() for w in [64, 256, 1024]}

    def test_init(self):
        stds = {}
        for w, model in self.models.items():
            for i, module in enumerate(list(model.modules())[1::2]):
                stds[(w, i+1, 'weight')] = module.weight.data.std()
                stds[(w, i+1, 'bias')] = module.bias.data.std()

        for w in [64, 256]:
            self.assertLess(
                torch.abs(
                    stds[(1024, 1, 'weight')] - stds[(w, 1, 'weight')]
                ) / stds[(1024, 1, 'weight')], 3e-3)
            # for l in [1, 2]:
            #     self.assertLess(
            #         torch.abs(
            #             stds[(1024, l, 'bias')] - stds[(w, l, 'bias')]
            #         ) / stds[(1024, l, 'bias')], 1e-1)
        self.assertTrue(
            stds[(1024, 2, 'weight')] < stds[(256, 2, 'weight')] < stds[(64, 2, 'weight')])
        for w in [64, 256, 1024]:
            self.assertEqual(stds[(w, 3, 'weight')], 0)
            self.assertEqual(stds[(w, 3, 'bias')], 0)
    
    def _test_train(self, opt):
        loss = {w: getattr(self, f'train_{opt}')(model, 201) for w, model in self.models.items()}
        with self.subTest(name=f'{opt}, step 1'):
            self.assertTrue(
                loss[64][0] == loss[256][0] == loss[1024][0],
                {k: v[0] for k, v in loss.items()})
        for t in [100, 200]:
            with self.subTest(name=f'{opt}, step {t+1}'):
                self.assertTrue(
                    loss[64][t] > loss[256][t] > loss[1024][t],
                    {k: v[t] for k, v in loss.items()})
    
    def test_sgd(self):
        self._test_train('sgd')

    def test_adam(self):
        self._test_train('adam')

class CNNTrainCase(unittest.TestCase):

    def train_adam(self, model, step):
        return train_model_MuAdam(model, train_loader, step=step, flatten_input=False)

    def train_sgd(self, model, step):
        return train_model_MuSGD(model, train_loader, step=step, flatten_input=False)

    def setUp(self):
        self.models = {w: generate_CNN(w, mup=True, bias=True, readout_zero_init=True, base_width=8, init='kaiming_fan_in_normal', bias_zero_init=False).cuda() for w in [8, 32, 128]}

    def test_init(self):
        stds = {}
        names = [0, 3, 7, 9, 11]
        for w, model in self.models.items():
            for i, module in enumerate(model):
                if i in names:
                    stds[(w, i, 'weight')] = module.weight.data.std()
                    stds[(w, i, 'bias')] = module.bias.data.std()

        for w in [8, 32]:
            self.assertLess(
                torch.abs(
                    stds[(128, 0, 'weight')] - stds[(128, 0, 'weight')]
                ) / stds[(128, 0, 'weight')], 3e-3)
        for name in names[:-1]:
            self.assertLess(
                torch.abs(
                    stds[(128, 0, 'bias')] - stds[(w, 0, 'bias')]
                ) / stds[(128, 0, 'bias')], 2e-1)
        for name in names[1:-1]:
            self.assertTrue(
                stds[(128, name, 'weight')] < stds[(32, name, 'weight')] < stds[(8, name, 'weight')])
        for w in [8, 32, 128]:
            self.assertEqual(stds[(w, 11, 'weight')], 0)
            self.assertEqual(stds[(w, 11, 'bias')], 0)
    
    def _test_train(self, opt):
        loss = {w: getattr(self, f'train_{opt}')(model, 201) for w, model in self.models.items()}
        with self.subTest(name=f'{opt}, step 1'):
            self.assertTrue(
                loss[8][0] == loss[32][0] == loss[128][0],
                {k: v[0] for k, v in loss.items()})
        for t in [200]:
            with self.subTest(name=f'{opt}, step {t+1}'):
                losses = {k: v[t] for k, v in loss.items()}
                # print(losses)
                self.assertTrue(
                    loss[8][t] > loss[32][t] > loss[128][t],
                    losses)
    
    def test_sgd(self):
        self._test_train('sgd')

    def test_adam(self):
        self._test_train('adam')

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(BackwardCompatibleCase))
    suite.addTests(unittest.makeSuite(MLPTrainCase))
    suite.addTests(unittest.makeSuite(CNNTrainCase))
    suite.addTests(unittest.makeSuite(CoordCheckCase))
    suite.addTests(unittest.makeSuite(SetBaseShapeCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=False)
    runner.run(suite())
