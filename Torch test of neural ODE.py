import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
print(use_cuda)

