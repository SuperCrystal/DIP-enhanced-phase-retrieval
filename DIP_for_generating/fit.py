# 网络训练
import math
import numpy as np
import torch
import torch.optim
from scipy.linalg import hadamard
from torch.autograd import Function, Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from util import data_transform

dtype = torch.cuda.FloatTensor

PHASE_RANGE = 0.5

# 两步菲涅尔衍射op，这里是利用pytorch的numpy扩展，似乎需要自己写backward，因为无法自动求导
# class TwoStepFresnel(torch.autograd.Function):
#     def forward(self, retrieved_phase, defocus=0):
#         # retrieved_phase为网络输出结果乘上2*pi*0.5
#         retrieved_phase = retrieved_phase.numpy()
#         wavelength = 632.8e-6
#         N = 256
#         width = 10
#         k = 2*np.pi/wavelength
#         f = 1000
#         z = 1000 + defocus
#         tmp = np.arange(-width//2, width//2, width/N)
#         x, y = np.meshgrid(tmp, tmp)
#         R = np.sqrt(x**2 + y**2)
#         R[R>width/2] = 0
#         lens = np.exp(-1j * k * R**2 / (2*f))* self._cyl(x,y,width/2)
#         u0 = lens * np.exp(1j*retrieved_phase)
#         uz = self._two_step_prop_fresnel(u0, wavelength, width/N, 1*width/N, z)
#         magnitudes = np.abs(uz)
#         return torch.Tensor(magnitudes)   # 输出振幅，和calc_intensity一致
#
#     def _cyl(self, x, y, r0):
#         r = np.sqrt(x**2 + y**2)
#         z = (r<r0) * 1.0
#         return z
#
#     def _ft2(self, g, delta):
#         return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
#
#     def _two_step_prop_fresnel(self, Uin, wvl, d1, d2, Dz):
#         N = Uin.shape[0]
#         k = 2 * np.pi / wvl
#
#         tmp = np.arange(-N//2, N//2)
#         x1, y1 = np.meshgrid(tmp*d1, tmp*d1)
#
#         m = d2 / d1
#         Dz1 = Dz / (1-m)
#         d1a = wvl * np.abs(Dz1) / (N * d1)
#         x1a, y1a = np.meshgrid(tmp*d1a, tmp*d1a)
#
#         Uitm = (1 / (1j*wvl*Dz1)
#                 * np.exp(1j*k/(2*Dz1)*(x1a**2+y1a**2))
#                 * self._ft2(Uin*np.exp(1j*k/(2*Dz1)*(x1**2+y1**2)), d1))
#
#         Dz2 = Dz - Dz1
#
#         x2, y2 = np.meshgrid(tmp*d2, tmp*d2)
#
#         Uout = (1 / (1j*wvl*Dz2)
#                 * np.exp(1j*k/(2*Dz2)*(x2**2+y2**2))
#                 * self.ft2(Uitm*np.exp(1j*k/(2*Dz2)*(x1a**2+y1a**2)), d1a))
#         return Uout
#
#     def backward(self, grad_output):
#         return grad_output

# torch版
# 根据官网信息 fft currently dont use complex tensors but the API will be soon updated
def ComplexMulti(a, b):
    # 输入shape (n, n, 2)
    # N = a.shape[0]
    # a = a.view(-1, 2)
    # b = b.view(-1, 2)
    # print(a.device, b.device)
    if len(a.shape) == 1:
        a = a.view(1, 1, 2)
    if len(b.shape) == 1:
        b = b.view(1, 1, 2)

    c1 = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    c1 = c1.unsqueeze(-1)
    c2 = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    c2 = c2.unsqueeze(-1)
    c = torch.cat([c1, c2], -1)

    return c


class CylBlock(torch.nn.Module):
    """
    自定义的圆形通光区域
    """
    def __init__(self, d1, d2, N):
        super(cyl, self).__init__()
        # tmp = torch.arange(-d1/2+d1/N, d1/2+d1/N, d1/N)
        tmp = torch.arange(-d1/2, d1/2+d1/(N-1), d1/(N-1))
        x, y = torch.meshgrid(tmp, tmp)
        R = torch.sqrt(x**2 + y**2)
        R = R.cuda()
        self.circle = (R<d2/2) * torch.cuda.FloatTensor([1.0])

    def forward(self, x):
        return torch.mul(x, self.circle)


class Ft2(torch.nn.Module):
    def __init__(self, delta):
        super(Ft2, self).__init__()
        self.delta = delta

    def forward(self, x):
        # return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
        y = torch.fft(x, signal_ndim=2) * self.delta**2
        return y   # 返回双通道表示的复数结果


class TwoStepProp(torch.nn.Module):
    def __init__(self, N, wvl, d1, d2, Dz):
        super(TwoStepProp, self).__init__()
        # self.N = Uin.shape[0]   # (w, h, 2)
        k = 2 * math.pi / wvl

        tmp = torch.arange(-N//2, N//2)  # N维向量
        self.d1 = d1
        self.x1, self.y1 = torch.meshgrid(tmp*self.d1, tmp*self.d1)

        m = d2 / self.d1
        self.Dz1 = Dz / (1-m)
        self.Dz2 = Dz - self.Dz1
        self.d1a = wvl * abs(self.Dz1) / (N * self.d1)
        self.x1a, self.y1a = torch.meshgrid(tmp*self.d1a, tmp*self.d1a)
        x2, y2 = torch.meshgrid(tmp*d2, tmp*d2)

        self.Uitm_1 = torch.Tensor([0, -wvl*self.Dz1]).type(dtype)
        self.Uitm_2 = torch.cat([torch.cos(k/(2*self.Dz1)*(self.x1a**2+self.y1a**2)).unsqueeze(-1), torch.sin(k/(2*self.Dz1)*(self.x1a**2+self.y1a**2)).unsqueeze(-1)], dim=-1).type(dtype)
        self.ft2_1 = Ft2()

        self.Uout_1 = torch.Tensor([0, -wvl*self.Dz2]).type(dtype)
        self.Uout_2 = torch.cat([torch.cos(k/(2*self.Dz2)*(self.x2**2+self.y2**2)).unsqueeze(-1), torch.sin(k/(2*self.Dz2)*(self.x2**2+self.y2**2)).unsqueeze(-1)], dim=-1).type(dtype)
        self.ft2_2 = Ft2()

    def forward(self, Uin):
        # Uin为双通道表示的复数
        # Uitm = (1 / (1j*wvl*Dz1)
        #         * torch.exp(1j*k/(2*Dz1)*(x1a**2+y1a**2))
        #         * self._ft2(Uin*np.exp(1j*k/(2*Dz1)*(x1**2+y1**2)), d1))
        Uitm_3_tmp = ComplexMulti(Uin,
                                    torch.cat([torch.cos(self.k/(2*self.Dz1)*(self.x1**2+self.y1**2)).unsqueeze(-1),
                                               torch.sin(self.k/(2*self.Dz1)*(self.x1**2+self.y1**2)).unsqueeze(-1)],
                                               dim=-1).type(dtype))
        Uitm_3 = self.ft2_1(Uitm_3_tmp, self.d1)  # exp还有问题
        Uitm = ComplexMulti(ComplexMulti(self.Uitm_1, self.Uitm_2), Uitm_3)  # (n, n, 2)

        # Uout = (1 / (1j*wvl*Dz2)
        #         * np.exp(1j*k/(2*Dz2)*(x2**2+y2**2))
        #         * self.ft2(Uitm*np.exp(1j*k/(2*Dz2)*(x1a**2+y1a**2)), d1a))
        Uout_3_tmp = ComplexMulti(Uitm,
                                    torch.cat([torch.cos(self.k/(2*self.Dz2)*(self.x1a**2+self.y1a**2)).unsqueeze(-1),
                                               torch.sin(self.k/(2*self.Dz2)*(self.x1a**2+self.y1a**2)).unsqueeze(-1)],
                                               dim=-1).type(dtype))
        Uout_3 = self.ft2_2(Uout_3_tmp, self.d1a)
        Uout = ComplexMulti(ComplexMulti(self.Uout_1, self.Uout_2), Uout_3)

        return Uout


class TwoStepFresnel(torch.nn.Module):
    def __init__(self, distance):
        super(TwoStepFresnel, self).__init__()
        # 常量使用数值
        wavelength = 632.991e-6
        d1 = 0.01       # 物面间隔
        d2 = 3.69e-3    # 像面间隔为像元大小
        N = 1024
        width = N * d1
        self.k = 2*math.pi/wavelength
        self.f = 250
        self.z = self.f + self.defocus
        self.tmp = torch.arange(-width/2, width/2, d1).type(dtype)
        x, y = torch.meshgrid(self.tmp, self.tmp)
        R = torch.sqrt(x**2 + y**2)
        R[R>width/2] = 0
        self.phase_lens = -self.k * R**2 / (2*self.f)

        self.two_step_prop = TwoStepProp(wavelength, d1, d2, distance)

    def forward(self, retrieved_phase):
        # retrieved_phase为网络输出结果乘上2*pi*0.5
        phase_u0 = self.phase_lens + retrieved_phase
        real_part = torch.cos(phase_u0).unsqueeze(-1)
        image_part = torch.sin(phase_u0).unsqueeze(-1)
        u0 = torch.cat((real_part, image_part), dim=-1).squeeze()  # 双通道表示的复数lens调制

        # u0 = lens * np.exp(1j*retrieved_phase)

        uz = self.two_step_prop(u0)  # (n, n, 2)

        re = torch.index_select(uz, dim=2, index=torch.tensor(0).type(torch.cuda.LongTensor))
        im = torch.index_select(uz, dim=2, index=torch.tensor(1).type(torch.cuda.LongTensor))
        out = torch.sqrt(re**2 + im**2).squeeze()
        # -- 归一化
        out = torch.square(out)
        out = out / torch.max(out)
        out = torch.sqrt(out)
        return out   # 输出振幅，和calc_intensity一致


# TV全变分loss
class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# 自定义的weighted mse loss
class WMSE(torch.nn.Module):
    def __init__(self, ratio=10, half_width=5):
        super(WMSE, self).__init__()
        self.ratio = ratio
        self.half_width = half_width

    def forward(self, output, gt):
        # 手动指定范围加权
        weight = torch.ones(output.shape).type(dtype) * torch.tensor(self.ratio).type(dtype)
        # self.half_width = 5
        weight[:self.half_width, :self.half_width] = 1.
        weight[:self.half_width, -self.half_width:] = 1.
        weight[-self.half_width:, :self.half_width] = 1.
        weight[-self.half_width:, -self.half_width:] = 1.
        # energe-weighted
        # weight = 1/ torch.sqrt(calc_intensity(output))
        # 归一化
        weight /= torch.sum(weight)  # torch tensor 相除是整数？？
        # print("the value of WMSE weight: %f  %f" % (weight[0][0], weight[128][128]))
        loss = torch.sum(torch.mul(torch.square((output-gt)), weight.type(dtype)))
        return loss

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, decay_weight=0.8):
    lr = init_lr * (decay_weight**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def cyl(input, d1, d2, N):
    # tmp = torch.arange(-d1/2+d1/N, d1/2+d1/N, d1/N)
    tmp = torch.arange(-d1/2, d1/2+d1/(N-1), d1/(N-1))
    x, y = torch.meshgrid(tmp, tmp)
    R = torch.sqrt(x**2 + y**2)
    R = R.cuda()
    circle = (R<d2/2) #* torch.cuda.FloatTensor([1.0])
    # print(circle.shape)
    circle = circle.view(1,1,N,N,1)
    return torch.mul(input, circle)


def calc_intensity(input):   # 其实计算的是振幅而非强度！
    # 此处是 exp(1j*input) input相当于是相位部分
    real_part = torch.cos(input).unsqueeze(-1)
    image_part = torch.sin(input).unsqueeze(-1)
    # print(real_part.shape)
    real_part = cyl(real_part, 34.31, 10, 1024)   # 振幅约束？？
    image_part = cyl(image_part,34.31, 10, 1024)
    # print(real_part.shape)
    complex_phase = torch.cat((real_part, image_part), dim=-1).squeeze()
    f_phase = torch.fft(complex_phase, signal_ndim=2)
    re = torch.index_select(f_phase, dim=2, index=torch.tensor(0).type(torch.cuda.LongTensor))
    im = torch.index_select(f_phase, dim=2, index=torch.tensor(1).type(torch.cuda.LongTensor))
    pred_intensity = torch.sqrt(re**2 + im**2).squeeze() # 这里算的是振幅
    # 根据实验情况 使用归一化光强
    pred_intensity = torch.square(pred_intensity) # 光强
    pred_intensity = pred_intensity / torch.max(pred_intensity) # 强度归一化
    pred_intensity = torch.sqrt(pred_intensity) # 振幅
    return pred_intensity

def fit(net,
        net_input=None,  # 是 measured intensity
        ref_intensity1=None,
        ref_intensity2=None,
        ref_intensity3=None,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        lr_decay_epoch=0,
        weight_decay=0,
        add_noise=False,
        gt=None,
        cosLR=False,
        reducedLR=False,
        modulate1=None,
        modulate2=None,
        modulate3=None,
        defocus_term=None):
    if net_input is not None:
        print('input provided')
    else:
        # 随机产生一个input
        print('error: input not defined!')
        return

    net_input_copy = net_input.data.clone()
    noise = net_input.data.clone()

    if OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)

    # 定义下onplateau学习率下降
    if reducedLR:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30)
    if cosLR:
        print('using cosine annealing lr')
        scheduler = CosineAnnealingLR(optimizer, T_max=500)

    # 定义损失函数计算方式
    mse = torch.nn.MSELoss()
    wmse = WMSE()
    tvloss = TVLoss(1000)
    # wmse = torch.nn.MSELoss()

    # ?
    mse_loss = np.zeros(num_iter)
    mse_loss2 = np.zeros(num_iter//10)

    # 用于保存loss最小的model
    min_loss = 10e9

    for i in range(num_iter):
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)

        if add_noise:
            # 给input加上随机噪声（均匀，0-1/30）辅助收敛，参考paper
            net_input = Variable(net_input_copy + noise.normal_()*1/30)

        def closure():
            # ------ old -------
            # # 梯度清零
            # optimizer.zero_grad()
            # # out = net(net_input.type(dtype)) * torch.tensor(2*np.pi*0.5)
            # out = net(net_input.type(dtype)) * torch.tensor(2*np.pi*PHASE_RANGE)
            # if modulate1 is not None:
            #     out1 = out + modulate1.cuda()   # 网络只需学习非调制部分
            # if modulate2 is not None:
            #     out2 = out + modulate2.cuda()
            # 这里的out是相位，分布在0-π之间
            # ------ old -------

            # ------- auto focus ---------
            # 梯度清零
            optimizer.zero_grad()
            out, out1, out2, out3, defocus_param1, defocus_param2, defocus_param3 = net(net_input.type(dtype), defocus_term=defocus_term)

            # ------- auto focus ---------


            # loss计算
            # 期望网络输出的是 phase，因此需要经过一个physical model
            # real_part = torch.cos(out).unsqueeze(-1)
            # image_part = torch.sin(out).unsqueeze(-1)
            # complex_phase = torch.cat((real_part, image_part), dim=-1).squeeze()
            # f_phase = torch.fft(complex_phase, signal_ndim=2)
            # re = torch.index_select(f_phase, dim=2, index=torch.tensor(0).type(torch.cuda.LongTensor))
            # im = torch.index_select(f_phase, dim=2, index=torch.tensor(1).type(torch.cuda.LongTensor))
            # pred_intensity = torch.sqrt(re**2 + im**2).squeeze()
            pred_intensity = calc_intensity(out)


            # 如果input是随机的而非intensity，则此处应该再传一个intensity的gt进来
            # loss = mse(pred_intensity, net_input.squeeze())
            # weighted loss 改善intensity的高能光斑拟合的好 但重要的细节拟合差
            loss = wmse(pred_intensity, net_input.squeeze())
            if modulate1 is not None:
                pred_intensity1 = calc_intensity(out1)
                loss += wmse(pred_intensity1, ref_intensity1.squeeze())
            if modulate2 is not None:
                pred_intensity2 = calc_intensity(out2)
                loss += wmse(pred_intensity2, ref_intensity2.squeeze())
            if modulate3 is not None:
                pred_intensity3 = calc_intensity(out3)
                loss += wmse(pred_intensity3, ref_intensity3.squeeze())

            # 全变分loss
            # loss += tvloss(out)

            # 还可以根据iteration，先按mse主要优化中心亮斑，再用wmse提升细节
            if reducedLR:
                scheduler.step(loss)
            if cosLR:
                scheduler.step()
            loss.backward()
            mse_loss[i] = loss.data.cpu().numpy()

            # 可以再将true phase传进来，观察迭代过程中拟合情况
            # 就像GS的误差曲线一样
            # true_loss =

            # 打印数值
            if i % 10 == 0:
                # out2 = net(Variable(net_intput).type(dtype))
                # loss2 = mse(torch.abs(torch.fft(out2, 2)), net_input)
                if gt is not None:
                    # out_2 = net(Variable(net_input).type(dtype)) * torch.tensor(2*np.pi*0.5)
                    phase_output, _, _, _, defocus_param1, defocus_param2, defocus_param3 = net(Variable(net_input).type(dtype), defocus_term.type(dtype))
                    loss2 = mse(phase_output, gt.type(dtype))
                    mse_loss2[i//10] = loss2.data.cpu().numpy()
                    # tvloss2 = tvloss(out2)
                    print('Iteration %05d    Train loss %f    phase diff mse %f    defocus_param1 %f    defocus_param2 %f    defocus_param3 %f'
                          % (i, loss.data, loss2.data, defocus_param1.data, defocus_param2.data, defocus_param3.data), '\r', end='')
                    # print('Iteration %05d    Train loss %f    phase diff mse %f     tv loss %f' % (i, loss.data, loss2.data, tvloss2.data), '\r', end='')
                else:
                    print('Iteration %05d    Train loss %f' % (i, loss.data), '\r', end='')

            return loss

        loss = optimizer.step(closure)

        # 保存loss最小、即恢复的最好的model
        if loss < min_loss:
            min_loss = loss.data
            torch.save(net.state_dict(), 'best.pth')

        # 保存最后一次迭代结果
        if i == num_iter - 1:
            torch.save(net.state_dict(), 'last.pth')

    return mse_loss, mse_loss2, net_input, net


def fit_exp(net,
        net_input=None,  # 是 measured intensity
        ref_intensity1=None,
        ref_intensity2=None,
        ref_intensity3=None,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        lr_decay_epoch=0,
        weight_decay=0,
        add_noise=False,
        gt=None,
        cosLR=False,
        reducedLR=False,
        defocus_term=None):
    if net_input is not None:
        print('input provided')
    else:
        # 随机产生一个input
        print('error: input not defined!')
        return

    net_input_copy = net_input.data.clone()
    noise = net_input.data.clone()

    if OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)

    # 定义下onplateau学习率下降
    if reducedLR:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1000)
    if cosLR:
        print('using cosine annealing lr')
        scheduler = CosineAnnealingLR(optimizer, T_max=500)

    # 定义损失函数计算方式
    mse = torch.nn.MSELoss()
    wmse = WMSE(1, 8)
    wmse1 = WMSE(1, 20)
    wmse2 = WMSE(1, 10)
    wmse3 = WMSE(1, 10)
    # tvloss = TVLoss(1000)
    # wmse = torch.nn.MSELoss()

    # ?
    mse_loss = np.zeros(num_iter)
    mse_loss2 = np.zeros(num_iter//10)

    # 用于保存loss最小的model
    min_loss = 10e9

    for i in range(num_iter):
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)

        if add_noise:
            # 给input加上随机噪声（均匀，0-1/30）辅助收敛，参考paper
            net_input = Variable(net_input_copy + noise.normal_()*1/30)

        def closure():
            # ------- auto focus ---------
            # 梯度清零
            optimizer.zero_grad()
            out, out1, out2, out3, defocus_param1, defocus_param2, defocus_param3 = net(net_input.type(dtype), defocus_term=defocus_term)

            # ------- auto focus ---------
            pred_intensity = calc_intensity(out)


            # 如果input是随机的而非intensity，则此处应该再传一个intensity的gt进来
            # loss = mse(pred_intensity, net_input.squeeze())
            # weighted loss 改善intensity的高能光斑拟合的好 但重要的细节拟合差
            loss = wmse(pred_intensity, net_input.squeeze())
            if ref_intensity1 is not None:
                pred_intensity1 = calc_intensity(out1)
                loss += wmse1(pred_intensity1, ref_intensity1.squeeze())
            if ref_intensity2 is not None:
                pred_intensity2 = calc_intensity(out2)
                loss += wmse2(pred_intensity2, ref_intensity2.squeeze())
            if ref_intensity3 is not None:
                pred_intensity3 = calc_intensity(out3)
                loss += wmse3(pred_intensity3, ref_intensity3.squeeze())

            # 全变分loss
            # loss += tvloss(out)

            # 还可以根据iteration，先按mse主要优化中心亮斑，再用wmse提升细节
            if reducedLR:
                scheduler.step(loss)
            if cosLR:
                scheduler.step()
            loss.backward()
            mse_loss[i] = loss.data.cpu().numpy()

            # 可以再将true phase传进来，观察迭代过程中拟合情况
            # 就像GS的误差曲线一样
            # true_loss =

            # 打印数值
            if i % 10 == 0:
                # out2 = net(Variable(net_intput).type(dtype))
                # loss2 = mse(torch.abs(torch.fft(out2, 2)), net_input)
                if gt is not None:
                    # out_2 = net(Variable(net_input).type(dtype)) * torch.tensor(2*np.pi*0.5)
                    phase_output, _, _, _, defocus_param1, defocus_param2, defocus_param3 = net(Variable(net_input).type(dtype), defocus_term.type(dtype))
                    loss2 = mse(phase_output, gt.type(dtype))
                    mse_loss2[i//10] = loss2.data.cpu().numpy()
                    # tvloss2 = tvloss(out2)
                    print('Iteration %05d    Train loss %f    phase diff mse %f    defocus_param1 %f    defocus_param2 %f    defocus_param3 %f'
                          % (i, loss.data, loss2.data, defocus_param1.data, defocus_param2.data, defocus_param3.data), '\r', end='')
                    # print('Iteration %05d    Train loss %f    phase diff mse %f     tv loss %f' % (i, loss.data, loss2.data, tvloss2.data), '\r', end='')
                else:
                    print('Iteration %05d    Train loss %f    defocus_param1 %f    defocus_param2 %f    defocus_param3 %f' % (i, loss.data, defocus_param1.data, defocus_param2.data, defocus_param3.data), '\r', end='')

            return loss

        loss = optimizer.step(closure)

        # 保存loss最小、即恢复的最好的model
        if loss < min_loss:
            min_loss = loss.data
            torch.save(net.state_dict(), 'best.pth')

        # 保存最后一次迭代结果
        if i == num_iter - 1:
            torch.save(net.state_dict(), 'last.pth')

    return mse_loss, mse_loss2, net_input, net


def fit_fresnel(net,
        net_input=None,  # 是 measured intensity
        ref_intensity1=None,
        ref_intensity2=None,
        ref_intensity3=None,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        lr_decay_epoch=0,
        weight_decay=0,
        add_noise=False,
        gt=None,
        cosLR=False,
        defocus1=None,
        defocus2=None):
    if net_input is not None:
        print('input provided')
    else:
        # 随机产生一个input
        print('error: input not defined!')
        return

    net_input_copy = net_input.data.clone()
    noise = net_input.data.clone()

    if OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)

    # 定义下onplateau学习率下降
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30)
    if cosLR:
        print('using cosine annealing lr')
        scheduler = CosineAnnealingLR(optimizer, T_max=500)

    # 定义损失函数计算方式
    mse = torch.nn.MSELoss()
    wmse = WMSE()
    tvloss = TVLoss(1000)
    prop1 = TwoStepProp(0)
    prop2 = TwoStepProp(4)
    prop3 = TwoStepProp(7)
    prop4 = TwoStepProp(10)
    # wmse = torch.nn.MSELoss()
    two_step_fresnel = TwoStepFresnel()

    # ?
    mse_loss = np.zeros(num_iter)
    mse_loss2 = np.zeros(num_iter//10)

    # 用于保存loss最小的model
    min_loss = 10e9

    for i in range(num_iter):
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)

        if add_noise:
            # 给input加上随机噪声（均匀，0-1/30）辅助收敛，参考paper
            net_input = Variable(net_input_copy + noise.normal_()*1/30)

        def closure():
            # 梯度清零
            optimizer.zero_grad()
            out = net(net_input.type(dtype)) * torch.tensor(2*np.pi*0.5)

            pred_intensity = two_step_fresnel(out, 0)

            loss = wmse(pred_intensity, net_input.squeeze())
            if defocus1 is not None:
                pred_intensity1 = two_step_fresnel(out, defocus1)
                loss += wmse(pred_intensity1, ref_intensity1.squeeze())
            if defocus2 is not None:
                pred_intensity2 = two_step_fresnel(out, defocus2)
                loss += wmse(pred_intensity2, ref_intensity2.squeeze())

            # 全变分loss
            # loss += tvloss(out)

            # 还可以根据iteration，先按mse主要优化中心亮斑，再用wmse提升细节
            # scheduler.step(loss)
            if cosLR:
                scheduler.step()
            loss.backward()
            mse_loss[i] = loss.data.cpu().numpy()

            # 可以再将true phase传进来，观察迭代过程中拟合情况
            # 就像GS的误差曲线一样
            # true_loss =

            # 打印数值
            if i % 10 == 0:
                # out2 = net(Variable(net_intput).type(dtype))
                # loss2 = mse(torch.abs(torch.fft(out2, 2)), net_input)
                if gt is not None:
                    out_2 = net(Variable(net_input).type(dtype)) * torch.tensor(2*np.pi*0.5)
                    loss2 = mse(out_2, gt.type(dtype))
                    mse_loss2[i//10] = loss2.data.cpu().numpy()
                    # tvloss2 = tvloss(out2)
                    print('Iteration %05d    Train loss %f    phase diff mse %f' % (i, loss.data, loss2.data), '\r', end='')
                    # print('Iteration %05d    Train loss %f    phase diff mse %f     tv loss %f' % (i, loss.data, loss2.data, tvloss2.data), '\r', end='')
                else:
                    print('Iteration %05d    Train loss %f' % (i, loss.data), '\r', end='')

            return loss

        loss = optimizer.step(closure)

        # 保存loss最小、即恢复的最好的model
        if loss < min_loss:
            min_loss = loss.data
            torch.save(net.state_dict(), 'best.pth')

        # 保存最后一次迭代结果
        if i == num_iter - 1:
            torch.save(net.state_dict(), 'last.pth')

    return mse_loss, mse_loss2, net_input, net
