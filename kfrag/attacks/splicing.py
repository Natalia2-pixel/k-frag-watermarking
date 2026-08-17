from .partial import overlay
def splice(destination,source,box=(.25,.25,.5,.5)): return overlay(destination,source,1.,box)
def collage(images):
    import torch
    if len(images)!=4: raise ValueError("collage requires four batches")
    top=torch.cat((images[0],images[1]),-1); bottom=torch.cat((images[2],images[3]),-1); return torch.cat((top,bottom),-2)
def fragment_replay(destination,source,region):
    row,col=divmod(region,4); out=destination.clone(); h,w=destination.shape[-2:]; out[...,row*h//4:(row+1)*h//4,col*w//4:(col+1)*w//4]=source[...,row*h//4:(row+1)*h//4,col*w//4:(col+1)*w//4]; return out
