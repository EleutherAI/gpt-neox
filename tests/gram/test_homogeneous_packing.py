"""Check exact token slices, mixed-document boundaries, and next-token targets."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import subprocess
import sys
import types

source=Path(__file__).resolve().parents[2]/'megatron/data/gram_packed_dataset.py'
spec=importlib.util.spec_from_file_location('gram_packed_dataset',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Corpus:
    docs=[np.array([10,11,12]),np.array([90,91,92,93]),np.array([],dtype=int),
          np.array([13,14,15,16,17,18]),np.array([94,95,96,97,98])]
    sizes=np.array([len(d) for d in docs])
    def get(self,index,offset=0,length=None):
        return self.docs[index][offset:] if length is None else self.docs[index][offset:offset+length]

def fixture(tmp_path):
    corpus=Corpus()
    for route,ids in enumerate(([0,2,3],[1,4])):
        np.save(tmp_path/f'doc_ids_{route}.npy',np.array(ids))
        np.save(tmp_path/f'offsets_{route}.npy',np.r_[0,np.cumsum(corpus.sizes[ids])])
    np.save(tmp_path/'sample_order.npy',np.array([3,0,2,1]))
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(seq_length=4,num_samples=4,
        documents=5,samples_per_route=[2,2])))
    return corpus,module.HomogeneousPackedDataset(corpus,tmp_path,4,4)

def test_every_token_and_next_target_belongs_to_selected_route(tmp_path):
    corpus,dataset=fixture(tmp_path)
    streams=[np.concatenate([corpus.docs[i] for i in ids]) for ids in ([0,2,3],[1,4])]
    for index,original in enumerate([3,0,2,1]):
        route=int(original>=2);local=original-route*2
        item=dataset[index]
        assert item['gram_flag']==route
        np.testing.assert_array_equal(item['text'],streams[route][local*4:local*4+5])
        assert np.all(item['text']<90) if route==0 else np.all(item['text']>=90)
    assert len(dataset)==4

def test_reject_missing_commit_and_incorrect_training_horizon(tmp_path):
    corpus,dataset=fixture(tmp_path)
    with pytest.raises(ValueError,match='horizon'):
        module.HomogeneousPackedDataset(corpus,tmp_path,4,5)
    with pytest.raises(ValueError,match='horizon'):
        module.HomogeneousPackedDataset(corpus,tmp_path,8,4)
    (tmp_path/'COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):
        module.HomogeneousPackedDataset(corpus,tmp_path,4,4)

def test_index_builder_is_deterministic_and_audits_real_mmap(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    indexed_source=repo/'megatron/data/indexed_dataset.py'
    stub=types.ModuleType('megatron');stub.print_rank_0=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,'megatron',stub)
    spec=importlib.util.spec_from_file_location('indexed_dataset',indexed_source)
    indexed=importlib.util.module_from_spec(spec);spec.loader.exec_module(indexed)
    prefix=tmp_path/'corpus'
    writer=indexed.MMapIndexedDatasetBuilder(str(prefix)+'.bin',dtype=np.uint16)
    for doc in Corpus.docs:
        writer.add_item(doc.astype(np.uint16));writer.end_document()
    writer.finalize(str(prefix)+'.idx')
    np.save(tmp_path/'labels.npy',np.array([False,True,False,False,True]))
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(data_path=str(prefix),gram_labels_path=str(tmp_path/'labels.npy'),
        train_iters=3,train_batch_size=2,seq_length=4,seed=1234)))
    outputs=[]
    for name in ('a','b'):
        output=tmp_path/name
        subprocess.run([sys.executable,str(repo/'tools/gram/prepare_homogeneous_packing.py'),
            '--repo',str(repo),'--config',str(config),'--output',str(output)],check=True)
        meta=json.loads((output/'COMPLETE.json').read_text())
        assert meta['mixed_label_sequences']==0 and meta['audit_sequences']==10000
        outputs.append(output)
    for name in ('sample_order.npy','doc_ids_0.npy','doc_ids_1.npy','offsets_0.npy','offsets_1.npy'):
        np.testing.assert_array_equal(np.load(outputs[0]/name),np.load(outputs[1]/name))
