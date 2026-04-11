class Attack:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def forward(self, images, labels=None, target_labels=None):
        raise NotImplementedError

    def __call__(self, images, labels=None, target_labels=None):
        return self.forward(images, labels, target_labels)
