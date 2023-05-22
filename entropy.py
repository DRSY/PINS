import torch
import torch.nn as nn

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax(dim=-1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy

if __name__ == "__main__":
    input = torch.tensor([0.0, 0.0])
    print(calc_entropy(input).item())
