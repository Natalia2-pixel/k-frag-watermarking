from __future__ import annotations
import torch
def crop(x,top=.1,left=.1,height=.8,width=.8):
    h,w=x.shape[-2:]; y0,x0=int(h*top),int(w*left); return x[...,y0:y0+int(h*height),x0:x0+int(w*width)]
def occlude(x,box=(.25,.25,.5,.5),value=0.):
    y,x0,h,w=box; out=x.clone(); H,W=x.shape[-2:]; out[...,int(y*H):int((y+h)*H),int(x0*W):int((x0+w)*W)]=value; return out
def overlay(x,other,alpha=.5,box=(.25,.25,.5,.5)):
    y,x0,h,w=box; out=x.clone(); H,W=x.shape[-2:]; sl=(slice(None),slice(None),slice(int(y*H),int((y+h)*H)),slice(int(x0*W),int((x0+w)*W))); out[sl]=out[sl]*(1-alpha)+other[sl]*alpha; return out
