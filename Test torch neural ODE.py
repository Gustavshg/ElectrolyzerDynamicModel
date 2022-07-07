import math
import time

import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def ode_solve(z0, t0, t1, f):
    """这里程序里面给了一个最简单的欧拉法常微分方程初值解法"""
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())  # 这个里面应该是torch tensor里面的操作
    h = (t1 - t0) / n_steps
    t = t0
    z = z0
    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z


class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """计算f, a*df/dz, a*df/dp, a*df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(
                0)  # unsqueeze就是给一个向量在0维增加一个维度，相当于是把所有的batch恢复成一个batch？
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.flatten())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None
# class ODEAdjoint(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, z0, t, flat_parameters, func):
#         assert isinstance(func, ODEF)  # 如果方程不是ODEF的话，就会报错
#         bs, *z_shape = z0.size()
#         time_len = t.size(0)
#
#         with torch.no_grad():
#             z = torch.zeros(time_len, bs, *z_shape).to(z0)
#             z[0] = z0
#             for i_t in range(time_len - 1):
#                 z0 = ode_solve(z0, t[i_t], t[i_t + 1], func)
#                 z[i_t + 1] = z0
#         ctx.func = func
#         ctx.save_for_backward(t, z.clone(), flat_parameters)
#         return z
#
#     @staticmethod
#     def backward(ctx, dLdz):
#         """dLdZ shape: time_len,batch_size,*z_shape"""
#         func = ctx.func
#         t, z, flat_parameters = ctx.saved_tensors
#         time_len, bs, *z_shape = z.size()
#         n_dim = np.prod(z_shape)
#         n_params = flat_parameters.size(0)
#
#         # Dynamics of augmented system to be calculated backwards in time
#         def augmented_dynamics(aug_z_i, t_i):
#             """
#             tensors here are temporal slices
#             t_i - is tensor with size: bs, 1
#             aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
#             """
#             z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim: 2 * n_dim]  # ignore parameters and time
#
#             # Unflatten z and a
#             z_i = z_i.view(bs, *z_shape)  # 相当于是reshape成这个形状
#             a = a.view(bs, *z_shape)
#             with torch.set_grad_enabled(True):
#                 t_i = t_i.detach().requires_grad_(True)  # 将这个参数从反向传播中剥离出来，不进行训练
#                 z_i = z_i.detach().requires_grad_(True)
#                 func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
#                 adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(
#                     z_i)  # 这里的to相当于是把z_i 的格式刷给新的变量
#                 adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
#                 adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)
#
#             # Flatten f and adfdz
#             func_eval = func_eval.view(bs, n_dim)
#             adfdz = adfdz.view(bs, n_dim)
#             return torch.cat((func_eval, -adfdz, -adfdp, - adfdt), dim=1)
#
#         dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
#         with torch.no_grad():
#             # Create placeholders for output gradients
#             # Prev computed backward adjoints to be adjusted by direct gradients
#             adj_z = torch.zeros(bs, n_dim).to(dLdz)
#             adj_p = torch.zeros(bs, n_params).to(dLdz)
#             # 和上面的z和参数不同，时间的梯度是在所有时刻都需要的
#             adj_t = torch.zeros(time_len, bs, 1).to(dLdz)
#
#             for i_t in range(time_len - 1, 0, -1):
#                 z_i = z[i_t]
#                 t_i = t[i_t]
#                 f_i = func(z_i, t_i)
#
#                 # 计算直接梯度
#                 dLdz_i = dLdz[i_t]
#                 dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]  # 计算两个矩阵的乘法
#
#                 # Adjusting adjoints with direct gradients
#                 adj_z += dLdz_i
#                 adj_t[i_t] = adj_t[i_t] - dLdt_i
#
#                 # 打包增强后变量
#                 aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)
#                 # 求解增强系统的反向问题
#                 aug_ans = ode_solve(aug_z, t_i, t[i_t - 1], augmented_dynamics)
#                 # 拆包求解后的反向增强系统
#                 adj_z[:] = aug_ans[:, n_dim: 2 * n_dim]
#                 adj_p[:] += aug_ans[:, 2 * n_dim: 2 * n_dim + n_params]
#                 adj_t[i_t - 1] = aug_ans[:, 2 * n_dim + n_params:]
#
#                 del aug_z, aug_ans
#             # 利用直接梯度计算0时刻的ajoint
#             # 计算直接梯度
#             dLdz_0 = dLdz[0]
#             dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
#
#             # 调整adjoints
#             adj_z += dLdz_0
#             adj_t[0] = adj_t[0] - dLdt_0
#
#         return adj_z.view(bs, *z_shape), adj_t, adj_p, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]


"""以上是Neural ODE的实现方法"""

"""以下是调用实例"""


class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)  # 输入是两纬，输出也是两维
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


class SpiralFunctionExample(LinearODEF):
    """这里是一个螺旋的方程示意图"""

    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))


class RandomLinearODEF(LinearODEF):
    """这里是需要我们拟合的线性微分方程"""

    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2) / 2.)


class TestODEF(ODEF):
    def __init__(self, A, B, x0):
        super(TestODEF, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)

    def forward(self, x, t):
        xTx0 = torch.sum(x * self.x0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(x - self.x0) + torch.sigmoid(- xTx0) * self.B(x + self.x0)
        return dxdt


class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant  # 我们的模型中究竟是时变系统还是时不变系统，还有待考虑，目前看起来是时变系统概率高一些
        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)  # 类似于relu函数，但是结果小于0时约为-1，而不是0
        # 这里的inplace意味着可以直接修改之前传递下来的tensor，不需要额外储存多余的变量

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)  # 直接在最后一个维度，也就是freature个数里面加入一个t

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)

        return out


def to_np(x):
    return x.detach().cpu().numpy()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 9)):
    plt.figure(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)  # 这里的b_i代表着什么还没太看懂
    if trajs is not None:
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
        if save is not None:
            plt.savefig(save)
    obs_np = to_np(obs[0])

    xmin = min(obs_np[:, 0, 0])
    ymin = min(obs_np[:, 0, 1])
    xmax = max(obs_np[:, 0, 0])
    ymax = max(obs_np[:, 0, 1])

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.show()


def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=100):
    # create data
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29 * 5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int32)
    index_np = np.hstack([index_np[:, None]])  # 等价于index_np = np.expand_dims(index_np,axis=1)
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])  # 这里的None相当于增加一个维度
    times = torch.from_numpy(times_np[:, :, None]).to(z0)  # 这里的None相当于增加一个维度
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.rand_like(obs) * 0.01  # 加入一个原始数据1%的噪声，最后的shape = 200，1，2

    # 获取随机时间分布下的轨迹
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32

    def create_batch():
        t0 = np.random.uniform(0, t_max - max_delta_time)  # 从0导最大值时间中间随机抽取一个时间
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)  # 随机赋予一个时间间隔

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # 训练神经网络微分方程
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=1E-1)
    for i in range(n_steps):
        obs_, ts_ = create_batch()  # 这里的形状分别是16，1，2；16，1，1
        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(
            z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)
            plot_trajectories(obs=[obs], times=[times], trajs=[z_p])
            clear_output(wait=True)


# 这个部分是对于螺旋线方程的拟合过程
test = 3
if test == 1:
    ode_true = NeuralODE(SpiralFunctionExample())
    ode_trained = NeuralODE(RandomLinearODEF())

    conduct_experiment(ode_true, ode_trained, 3000, name='linear')

# 接下来是求解上面搞不清楚是什么的方程
if test == 2:
    A = Tensor([[-0.1, -0.5], [0.5, -0.1]])
    B = Tensor([[0.2, 1.0], [-1.0, 0.2]])
    x0 = Tensor([[-1.0, 0.0]])
    func = TestODEF(A, B, x0)
    ode_true = NeuralODE(func)
    func = NNODEF(2, 16, time_invariant=True)
    ode_trained = NeuralODE(func)
    conduct_experiment(ode_true, ode_trained, 3000, "comp", plot_freq=300)

"""这一部分是定义一个手写数字识别的神经网络并且进行训练"""


def norm(dim):
    # 这里是进行一个batch归一化，从而更好地控制层的输出
    return nn.BatchNorm2d(dim)


def conv3x3(in_feats, out_feats, stride=1):
    # 自己定义了一个核大小为3的卷积神经网络
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    """这里应该是吧时间变量加入到输入的tensor中，至于为什么要这么做，我们还是不太理解"""
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class ConvODEF(ODEF):
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):

        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt


class ContinuousNeuralMNISTClassifier(nn.Module):
    def __init__(self, ode):
        super(ContinuousNeuralMNISTClassifier, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3,1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2,1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)# x.shape = [32,64,6,6]
        x = self.feature(x)# x.shape = [32,64,6,6]，这里t自动赋值0~1
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()  # 还不太清楚这里的shape是什么
        x = x.view(-1, shape)
        out = self.fc(x)
        return out


func = ConvODEF(64)
ode = NeuralODE(func)
model = ContinuousNeuralMNISTClassifier(ode)

if use_cuda:
    model = model.cuda()
import torchvision

img_std = 0.3081
img_mean = 0.1307

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((img_mean,), (img_std,))
                               ])
                               ),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((img_mean,), (img_std,))
                               ])
                               ),
    batch_size=128, shuffle=True
)

optimizer = torch.optim.Adam(model.parameters())


def train(epoch):
    num_items = 0
    train_losses = []

    model.train()
    criterion = nn.CrossEntropyLoss()
    print(f"Training Epoch {epoch}...")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_losses += [loss.item()]
        num_items += data.shape[0]
    print('Train loss: {:.5f}'.format(np.mean(train_losses)))
    return train_losses


def test():
    accuracy = 0.0
    num_items = 0

    model.eval()  # 不启用模型中的batch normalization & dropout
    criterion = nn.CrossEntropyLoss()
    print(f"testing")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()  # 求出最大值所在的索引，并且和目标进行对应然后加和
            num_items += data.shape[0]
    accuracy = accuracy * 100 / num_items
    print("Test Accuracy:{:.3f}".format(accuracy))

n_epochs = 5
# test()
train_losses = []

for epoch in range(1,n_epochs+1):
    t0 = time.time()
    train_losses += train(epoch)
    print(time.time()-t0)
    test()

import pandas as pd

plt.figure(figsize=(9, 5))
history = pd.DataFrame({"loss": train_losses})
history["cum_data"] = history.index * batch_size
history["smooth_loss"] = history.loss.ewm(halflife=10).mean()
history.plot(x="cum_data", y="smooth_loss", figsize=(12, 5), title="train error")