"""Grouped routing must preserve data, route decisions and mean-example gradients."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from test_routing import Network, args, gram


class Timers:
    def __call__(self, name): return self
    def start(self): pass
    def stop(self): pass


class Engine:
    def __init__(self, module, route, gas):
        self.module=module;module._gram_controller=route
        self.optimizer=gram.RoutedAdamW(module.parameters(),route,lr=0.001,weight_decay=0.1,managed_step=True)
        self.gas=gas;self.micro_steps=0;self.boundary=None;self.updates=0;self.device=torch.device('cpu')
        self.saved_grads=None
    def set_gradient_accumulation_boundary(self, state): self.boundary=state
    def backward(self, loss): (loss/self.gas).backward()
    def step(self):
        self.micro_steps+=1
        if self.boundary:
            self.saved_grads=[p.grad.clone()for p in self.module.parameters()]
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.updates+=1


def forward(neox_args, timers, data_iterator, model, **kw):
    d=next(data_iterator);flag=int(d['gram_flag'][0])
    model.module._gram_controller.select(flag,neox_args.gram_core_robustness,neox_args.seed,
                                        neox_args.iteration,neox_args._gram_microstep)
    values=model.module(d['text']).square().mean(dim=-1)
    # Different valid-token counts exercise per-example rather than pooled-token means.
    mask=d['mask'];loss=((values*mask).sum(dim=1)/mask.sum(dim=1)).mean()
    return loss,{}


def backward(neox_args, timers, optimizer, model, loss): model.backward(loss)


def setup(inplace=False):
    torch.manual_seed(7)
    m=Network();c=args(gram_nested=True,gram_inner_labels_path='weak.npy',gram_optimizer_layout=3,
                       gram_grouped_microbatch_size=4,gram_inplace_mask=inplace,
                       gradient_accumulation_steps=16,seed=1234,iteration=3)
    route=gram.install(m,c);return Engine(m,route,16),c


def batches():
    out=[]
    for i,label in enumerate([0,0,1,0,2,0,0,1,0,0,2,0,1,0,0,0]):
        out.append(dict(text=torch.tensor([[i%12,1,2,3]]),sample_id=torch.tensor([i]),gram_flag=torch.tensor([label]),
                        mask=torch.tensor([[1.,1.,float(i%2),float(i%3==0)]])))
    return out


@pytest.mark.parametrize('inplace',[False,True])
@pytest.mark.parametrize('group_size',[2,3,4,6,8])
@pytest.mark.parametrize('clean_only',[False,True])
def test_grouped_update_matches_singleton_mean_and_optimizer_state(inplace,group_size,clean_only):
    reference,c=setup(inplace);grouped,gc=setup(inplace)
    gc.gram_grouped_microbatch_size=group_size
    for update in range(2):
        c.iteration=gc.iteration=3+update
        data=batches();ref_loss=0
        if clean_only:
            c.gram_core_robustness=gc.gram_core_robustness=0
            for d in data:d['gram_flag'].zero_()
        for i,d in enumerate(data):
            c._gram_microstep=i
            loss,_=forward(c,Timers(),iter([d]),reference)
            reference.backward(loss);ref_loss+=loss.detach()/16
        reference.module._gram_controller.prepare_step(reference.device)
        reference.set_gradient_accumulation_boundary(True);reference.step()
        reference.module._gram_controller.finish_step()
        result,skipped=gram.grouped_train_step(gc,Timers(),iter(data),grouped,grouped.optimizer,None,None,
                                              forward,backward,lambda xs:torch.stack(xs))
        assert skipped==0 and grouped.updates==update+1 and grouped.micro_steps==16*(update+1)
        assert grouped.boundary is None
        torch.testing.assert_close(result['lm_loss'],ref_loss,atol=1e-7,rtol=1e-6)
        for a,b in zip(reference.saved_grads,grouped.saved_grads):
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
        for a,b in zip(reference.module.parameters(),grouped.module.parameters()):
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
            for key in reference.optimizer.state[a]:
                torch.testing.assert_close(reference.optimizer.state[a][key],grouped.optimizer.state[b][key],atol=2e-7,rtol=2e-5)


def test_group_plan_contains_every_example_once_and_same_complete_routing_state():
    _,c=setup();data=batches();groups=gram.grouped_batches(data,c)
    assert sum(count for _,count,_ in groups)==len(data)
    assert all(1<=count<=4 for _,count,_ in groups)
    original=[]
    for i,d in enumerate(data):
        route=gram.RoutingController(nested=True);route.select(int(d['gram_flag'][0]),c.gram_core_robustness,c.seed,c.iteration,i)
        original.append((tuple(route.train_roles),route.aux_on,route.inner_on))
    consumed=[]
    # text[0] encodes i%12; compare the full text+mask/label identity as a multiset.
    for rep,count,d in groups:
        route=gram.RoutingController(nested=True);route.select(int(d['gram_flag'][0]),c.gram_core_robustness,c.seed,c.iteration,rep)
        state=(tuple(route.train_roles),route.aux_on,route.inner_on)
        assert state==original[rep]
        assert all(original[index]==state for index in d['sample_id'].tolist())
        consumed.extend((tuple(t.tolist()),tuple(m.tolist()),int(l))for t,m,l in zip(d['text'],d['mask'],d['gram_flag']))
    expected=[(tuple(d['text'][0].tolist()),tuple(d['mask'][0].tolist()),int(d['gram_flag'][0]))for d in data]
    assert sorted(consumed)==sorted(expected)


def test_absorbed_channels_preserve_exact_dense_mlp_parameters():
    dense=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,8))
    width=gram.absorbed_mlp_width(32,4,2,8)
    core=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,8))
    # AuxiliaryMLP construction doesn't call core.forward; parameter accounting is independent.
    wrapped=gram.AuxiliaryMLP(core,8,4,gram.RoutingController(nested=True),0.02,output_bias=False)
    assert sum(p.numel()for p in wrapped.parameters())==sum(p.numel()for p in dense.parameters())
    assert wrapped.gram_aux[2].bias is None and wrapped.gram_inner[2].bias is None
    assert gram.absorbed_mlp_width(16384,1024,2,512)==14336
    with pytest.raises(ValueError):gram.absorbed_mlp_width(256,1024,2,512)
