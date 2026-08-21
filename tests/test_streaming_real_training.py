import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from kfrag.data.manifests import assert_disjoint, report_manifest
from kfrag.training.trainer import _images, fresh_payloads
from scripts.train_kfrag import make_loader, portable_config, resolve_data


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, count=7): self.count=count; self.reads=[]
    def __len__(self): return self.count
    def __getitem__(self, index):
        self.reads.append(index)
        return {"image":torch.full((3,4,4),float(index)),"relative_id":str(index)}


def test_training_script_never_stacks_the_complete_dataset():
    source=(Path(__file__).parents[1]/"scripts"/"train_kfrag.py").read_text(encoding="utf-8")
    assert "torch.stack" not in source


def test_loader_materializes_only_the_requested_minibatch_and_is_seeded():
    first=CountingDataset(); loader=make_loader(first,2,0,True,torch.device("cpu"),91)
    batch=next(iter(loader)); assert len(batch["image"])==2 and len(first.reads)==2
    second=CountingDataset(); other=make_loader(second,2,0,True,torch.device("cpu"),91)
    assert batch["relative_id"]==next(iter(other))["relative_id"]
    assert loader.drop_last is False and loader.pin_memory is False


def test_split_overlap_is_rejected():
    try: assert_disjoint({"train":["same.jpg"],"validation":["same.jpg"],"test":[]})
    except ValueError as exc: assert "overlap" in str(exc)
    else: raise AssertionError("overlap accepted")


def test_payload_stream_advances_across_epochs_for_same_image():
    generator=torch.Generator().manual_seed(7)
    assert not torch.equal(fresh_payloads(1,generator=generator),fresh_payloads(1,generator=generator))


def _args(root,train_max=3,val_max=2):
    return argparse.Namespace(smoke=False,train_data_root=root,val_data_root=root,train_manifest=None,val_manifest=None,max_train_images=train_max,max_val_images=val_max)


def test_image_caps_and_empty_or_missing_roots(tmp_path):
    root=tmp_path/"images"; root.mkdir()
    for index in range(10): Image.new("RGB",(8,8),(index,0,0)).save(root/f"{index}.png")
    train,val,*_=resolve_data({"image_size":8},_args(root),3)
    assert len(train)==3 and len(val)==2
    missing=tmp_path/"missing"
    try: resolve_data({"image_size":8},_args(missing),3)
    except FileNotFoundError as exc: assert "dataset root does not exist" in str(exc)
    else: raise AssertionError("missing root accepted")
    empty=tmp_path/"empty"; empty.mkdir()
    try: resolve_data({"image_size":8},_args(empty),3)
    except RuntimeError as exc: assert "no images selected" in str(exc)
    else: raise AssertionError("empty root accepted")


def test_runtime_absolute_paths_are_not_serialized(tmp_path):
    safe=portable_config({"data_root":str(tmp_path),"nested":{"checkpoint":str(tmp_path/"a.pt")},"batch_size":2})
    assert str(tmp_path) not in json.dumps(safe) and safe["nested"]["checkpoint"]=="<runtime-path>"
    report=report_manifest("ImageFolderAdapter","train",["nested/a.jpg"],{"resize":256})
    assert set(report)=={"dataset_adapter","relative_image_identifiers","manifest_hash","image_count","split_name","deterministic_preprocessing"}


class MovementSpy:
    def __init__(self): self.calls=[]
    def to(self,*args,**kwargs): self.calls.append((args,kwargs)); return self


def test_cpu_and_cuda_movement_is_batch_wise_and_pin_memory_is_cuda_only():
    for name in ("cpu","cuda"):
        spy=MovementSpy(); result=_images({"image":spy},torch.device(name)); assert result is spy
        assert spy.calls==[((torch.device(name),),{"non_blocking":name=="cuda"})]
    dataset=CountingDataset(2)
    assert make_loader(dataset,1,0,False,torch.device("cuda"),1).pin_memory is True
