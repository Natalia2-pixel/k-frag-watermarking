import inspect, torch
from kfrag.models.learned_channel_v3 import ResidualSymbolSystem, BlindSymbolDecoder

def test_v3_is_blind_local_and_payload_sensitive():
    assert list(inspect.signature(BlindSymbolDecoder.forward).parameters)==["self","questioned_image"]
    model=ResidualSymbolSystem(); image=torch.full((2,3,64,64),.5); a=torch.zeros(2,8,4,4); b=1-a
    oa=model(image,a); ob=model(image,b)
    assert oa["symbol_logits"].shape==(2,8,4,4) and (oa["residual"]-ob["residual"]).abs().mean()>0
    assert (oa["symbol_logits"]<0).all() and (ob["symbol_logits"]>0).all()

def test_v3_residual_is_bounded_and_unsaturated():
    model=ResidualSymbolSystem(); out=model(torch.full((2,3,64,64),.5),torch.randint(0,2,(2,8,4,4)).float())
    assert out["residual"].abs().max()<=model.encoder.alpha
    assert out["residual"].abs().ge(model.encoder.alpha*.999).float().mean()<.001
