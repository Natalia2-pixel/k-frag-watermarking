from __future__ import annotations
import math, torch
def communication_metrics(logits,targets,symbol_slice=slice(4,12)):
    pred=logits>=0; correct=pred.eq(targets.bool()); prob=torch.sigmoid(logits); p=prob.mean((0,2,3)); entropy=-(p.clamp(1e-7,1-1e-7)*p.clamp(1e-7,1-1e-7).log2()+(1-p).clamp(1e-7,1).log2()*(1-p))
    symbol=correct[:,symbol_slice]
    return {"bit_accuracy":float(correct.float().mean().detach()),"per_bit_accuracy":[float(x.detach()) for x in correct.float().mean((0,2,3))],"exact_symbol_accuracy":float(symbol.all(1).float().mean().detach()),"exact_tag_accuracy":None if correct.shape[1]<=12 else float(correct[:,12:].all(1).float().mean().detach()),"exact_packet_accuracy":float(correct.all(1).float().mean().detach()),"entropy":float(entropy.mean().detach()),"confidence":float((prob-.5).abs().mul(2).mean().detach()),"predicted_one_frequency":float(pred.float().mean().detach()),"payload_sensitivity":None}
def fidelity_metrics(original,watermarked,residual,residual_alpha=None):
    mse=float((original-watermarked).square().mean().detach()); psnr=None if mse==0 else 10*math.log10(1/mse); mx,my=original.mean(),watermarked.mean(); vx,vy=original.var(unbiased=False),watermarked.var(unbiased=False); cov=((original-mx)*(watermarked-my)).mean(); ssim=float(((2*mx*my+1e-4)*(2*cov+9e-4)/((mx*mx+my*my+1e-4)*(vx+vy+9e-4))).detach())
    saturation=None if residual_alpha is None else float(residual.abs().ge(float(residual_alpha)*.999).float().mean().detach())
    return {"psnr":psnr,"ssim":ssim,"lpips":None,"maximum_residual":float(residual.abs().max().detach()),"mean_absolute_residual":float(residual.abs().mean().detach()),"saturation_fraction":saturation}
