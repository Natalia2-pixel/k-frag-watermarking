def publication_figure(path,original,watermarked,residual,questioned,protocol_map,manipulation_ground_truth=None,predicted_manipulation=None,metrics=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8)); imgs=[original,watermarked,(residual*.5/max(float(residual.abs().max().detach()),1e-8)+.5).clamp(0,1),questioned]
    titles=["Original","Watermarked","Amplified signed residual","Questioned image"]
    for ax,img,title in zip(axes[0],imgs,titles): ax.imshow(img[0].detach().cpu().permute(1,2,0)); ax.set_title(title); ax.axis("off")
    state_ids={s:i for i,s in enumerate(("missing_or_unobserved","undecodable","invalid_authentication","duplicate_or_conflicting","valid_authenticated"))}; matrix=[[state_ids[x] for x in row] for row in protocol_map]
    axes[1,0].imshow(matrix,vmin=0,vmax=4,cmap="viridis"); axes[1,0].set_title("Protocol evidence (authenticated states)")
    for ax,title,data in ((axes[1,1],"Manipulation ground truth",manipulation_ground_truth),(axes[1,2],"Predicted manipulation",predicted_manipulation)):
        ax.set_title(title); ax.imshow(data if data is not None else [[0]*4]*4,cmap="magma")
    axes[1,3].axis("off"); axes[1,3].text(0,.9,"\n".join(f"{k}: {v}" for k,v in (metrics or {}).items()),va="top")
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
