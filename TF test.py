import torch

cc=torch.randn((2,2,5))
dd=torch.randn((2,5,2))
print(torch.bmm(cc,dd))

