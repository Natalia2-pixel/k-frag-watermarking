from __future__ import annotations
import torch
from torch.nn import functional as F
def kfrag_losses(outputs,targets,image,alternate_residual=None,weights=None):
    weights={"packet":1.,"symbol":1.,"fidelity":.1,"sensitivity":.1,"balance":.01,**(weights or {})}; logits=outputs["packet_logits"]
    packet=F.binary_cross_entropy_with_logits(logits,targets); symbol=F.binary_cross_entropy_with_logits(logits[:,4:12]*2,targets[:,4:12]); fidelity=F.mse_loss(outputs["watermarked_image"],image)
    sensitivity=torch.zeros((),device=image.device) if alternate_residual is None else 1/(alternate_residual.sub(outputs["residual"]).abs().mean()+1e-6)
    region_energy=outputs["residual"].abs().mean((1,2,3)); balance=region_energy.var(unbiased=False)
    total=weights["packet"]*packet+weights["symbol"]*symbol+weights["fidelity"]*fidelity+weights["sensitivity"]*sensitivity+weights["balance"]*balance
    return {"total":total,"packet":packet,"symbol":symbol,"fidelity":fidelity,"sensitivity":sensitivity,"balance":balance}
