class RegenerationAdapter:
    def __init__(self,callable_=None): self.callable=callable_
    def __call__(self,image):
        if self.callable is None: raise RuntimeError("regeneration adapter is not configured")
        return self.callable(image)
