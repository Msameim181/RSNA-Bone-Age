import torch



def reload_model(net, path: str):
    net.load_state_dict(torch.load(path))
