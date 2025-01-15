import torch


def load_model(model_path):
    raise NotImplementedError

class EnsembleModel(torch.nn.Module):

    def __init__(self, *model_paths):
        super().__init__()
        self.models = []
        for model_path in model_paths:
            self.models.append(load_model(model_path))

    def forward(self, *inputs):
        out, *other_outputs = [model(*inputs) for model in self.models]
        for output in other_outputs:
            out += output
        return out


