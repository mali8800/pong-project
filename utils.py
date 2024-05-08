import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=get_device()).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
