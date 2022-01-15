import numpy as np
import torch
import random
import os


def seed_everything(seed=63):
    random.seed(seed)
    t_seed = random.randint(1, 1e6)
    np_seed = random.randint(1, 1e6)
    os_py_seed = random.randint(1, 1e6)

    np.random.seed(np_seed)
    torch.manual_seed(t_seed)
    os.environ["PYTHONHASHSEED"] = str(os_py_seed)

    if torch.cuda.is_available():
        tc_seed = random.randint(1, 1e6)
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(tc_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
