import torch
import torch.nn.functional as F
import numpy as np
from math import exp
import lpips
import math


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class LPIPS_Loss(torch.nn.Module):
    def __init__(self, net_str='alex'):
        super(LPIPS_Loss, self).__init__()
        self.lpips = lpips.LPIPS(net=net_str)  # best forward scores

    def forward(self, img0, img1):
        return self.lpips(img0, img1)


class LPIPS_L1_Loss(torch.nn.Module):
    def __init__(self, net_str='alex', lpips=0.1):
        super(LPIPS_L1_Loss, self).__init__()
        self.lpips = LPIPS_Loss(net_str=net_str)
        self.l1 = torch.nn.L1Loss()
        self.lambda_lpips = lpips

    def forward(self, img0, img1):
        return torch.mean(self.lpips(img0, img1)) * self.lambda_lpips + self.l1(img0, img1)


def avg_endpoint_error(input_flow, target_flow):
    diss = input_flow - target_flow
    diss_norm = torch.norm(diss, dim=1)
    return torch.mean(diss_norm)


def edge_ware_smoothing(flow, img):
    """
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_flow = flow.mean(2, True).mean(3, True)
    normed_flow = flow / (mean_flow + 1e-7)

    grad_flow_x = torch.abs(normed_flow[:, :, :, :-1] - normed_flow[:, :, :, 1:])
    grad_flow_y = torch.abs(normed_flow[:, :, :-1, :] - normed_flow[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_flow_x *= torch.exp(-grad_img_x)
    grad_flow_y *= torch.exp(-grad_img_y)

    return grad_flow_x.mean() + grad_flow_y.mean()


def sum_of_suppressed_accumulations(warped_events, p):
    """
    $$ sun_{i,j} e^{-h(i,j) * p}   $$
    :param warped_events: (B, 2, H, W)
    :return:
    """
    return torch.mean(torch.exp(-p * warped_events))


def sum_of_squares(warped_events):
    """
    $$ sum_{i,j} h(i,j)^2   $$
    :param warped_events: (B, 2, H, W)
    :return:
    """
    return torch.mean(torch.pow(warped_events, 2))


def total_variation(flow):
    """
    :param flow: (B, 2, H, W)
    :return:
    """
    B, _, h, w = flow.size()
    h_tv = torch.pow((flow[:, :, 1:, :] - flow[:, :, :h-1, :]), 2)
    w_tv = torch.pow((flow[:, :, :, 1:] - flow[:, :, :, :w-1]), 2)
    return torch.mean(h_tv) + torch.mean(w_tv)


def total_variation_sparse(flow):
    """
    :param flow: (B, 2, H, W)
    :return:
    """
    B, _, h, w = flow.size()
    h_tv = torch.abs((flow[:, :, 1:, :] - flow[:, :, :h-1, :]))
    w_tv = torch.abs((flow[:, :, :, 1:] - flow[:, :, :, :w-1]))
    return torch.mean(h_tv) + torch.mean(w_tv)


def magnitude(flow):
    """
    :param flow: (B, 2, H, W)
    :return:
    """
    B, _, h, w = flow.size()
    mag = torch.norm(flow, dim=1)
    return torch.mean(mag)


def pyramid_total_variation(flow, pyramid_depth=5):
    """
    :param flow: (B, 2, H, W)
    :return:
    """
    pyramid_0 = flow
    pyramid_1 = F.interpolate(flow, scale_factor=1/2, mode='bilinear')
    pyramid_2 = F.interpolate(flow, scale_factor=1/4, mode='bilinear')
    pyramid_3 = F.interpolate(flow, scale_factor=1/8, mode='bilinear')
    pyramid_4 = F.interpolate(flow, scale_factor=1/16, mode='bilinear')
    return total_variation(pyramid_0) + \
           total_variation(pyramid_1) + \
           total_variation(pyramid_2) + \
           total_variation(pyramid_3) + \
           total_variation(pyramid_4)


class LowFrequencySupervisionLoss(torch.nn.Module):
    def __init__(self, window_size=25, sigma=6):
        super(LowFrequencySupervisionLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.window = create_window(window_size, sigma)
        self.window_size = window_size

    def forward(self, img0, img1):
        B, C, H, W = img0.shape
        img0_blur = F.conv2d(img0, self.window.type_as(img0), padding=self.window_size // 2, groups=C)
        img1_blur = F.conv2d(img1, self.window.type_as(img1), padding=self.window_size // 2, groups=C)
        return self.l1(img0_blur, img1_blur)


def gradient(img_tensor):
    """
    :param img_tensor:
    :return:
    """
    B, _, h, w = img_tensor.size()
    h_tv = torch.pow((img_tensor[:, :, 1:, :] - img_tensor[:, :, :h-1, :]), 2)
    w_tv = torch.pow((img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :w-1]), 2)
    grad = torch.sqrt(h_tv[:, :, :, 1:] + w_tv[:, :, 1:, :])
    grad = F.pad(grad, (1, 0, 1, 0))
    return grad


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2/float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        window = self.window.type_as(img1)
        return _ssim(img1, img2, window, self.window_size, self.channel, self.size_average)


class GradSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(GradSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        window = self.window.type_as(img1)
        return _ssim(gradient(img1), gradient(img2), window, self.window_size, self.channel, self.size_average)


class SSIM_L1(torch.nn.Module):
    def __init__(self, alpha=0.7, window_size=11, channel=1, size_average=True):
        super(SSIM_L1, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.alpha = alpha
        self.window = create_window(window_size, self.channel)
        self.smoothl1 = torch.nn.L1Loss()

    def forward(self, img1, img2):
        window = self.window.type_as(img1)
        ssim_12 = _ssim(img1, img2, window, self.window_size, self.channel, self.size_average)
        return (self.alpha / 2.) * (1 - ssim_12) + (1 - self.alpha) * self.smoothl1(img1, img2)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def histogram_match(input, target, patch, stride):
    n1, c1, h1, w1 = input.size()
    n2, c2, h2, w2 = target.size()
    input.resize_(h1 * w1 * h2 * w2)
    target.resize_(h2 * w2 * h2 * w2)
    conv = torch.tensor((), dtype=torch.float32)
    conv = conv.new_zeros((h1 * w1, h2 * w2))
    conv.resize_(h1 * w1*h2 * w2)
    assert c1 == c2, 'input:c{} is not equal to target:c{}'.format(c1, c2)

    size1 = h1 * w1
    size2 = h2 * w2
    N = h1 * w1 * h2 * w2
    print('N is', N)

    for i in range(0, N):
        i1 = i / size2
        i2 = i % size2
        x1 = i1 % w1
        y1 = i1 / w1
        x2 = i2 % w2
        y2 = i2 / w2
        kernal_radius = int((patch - 1) / 2)

        conv_result = 0
        norm1 = 0
        norm2 = 0
        dy = -kernal_radius
        dx = -kernal_radius
        while dy <= kernal_radius:
            while dx <= kernal_radius:
                xx1 = x1 + dx
                yy1 = y1 + dy
                xx2 = x2 + dx
                yy2 = y2 + dy
                if 0 <= xx1 < w1 and 0 <= yy1 < h1 and 0 <= xx2 < w2 and 0 <= yy2 < h2:
                    _i1 = yy1 * w1 + xx1
                    _i2 = yy2 * w2 + xx2
                    for c in range(0, c1):
                        term1 = input[int(c * size1 + _i1)]
                        term2 = target[int(c * size2 + _i2)]
                        conv_result += term1 * term2
                        norm1 += term1 * term1
                        norm2 += term2 * term2
                dx += stride
            dy += stride
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        conv[i] = conv_result / (norm1 * norm2 + 1e-9)

    match = torch.tensor((), dtype=torch.float32)
    match = match.new_zeros(input.size())

    correspondence = torch.tensor((), dtype=torch.int16)
    correspondence.new_zeros((h1, w1, 2))
    correspondence.resize_(h1*w1*2)

    for id1 in range(0, size1):
        conv_max = -1e20
        for y2 in range(0, h2):
            for x2 in range(0, w2):
                id2 = y2 * w2 + x2
                id = id1 * size2 + id2
                conv_result = conv[id1]

                if conv_result > conv_max:
                    conv_max = conv_result
                    correspondence[id1 * 2 + 0] = x2
                    correspondence[id1 * 2 + 1] = y2

                    for c in range(0, c1):
                        match[c * size1 + id1] = target[c * size2 + id2]

    match.resize_((n1, c1, h1, w1))

    return match, correspondence


def RMSE(disp0, dispgt, mask):
    rmse = (dispgt * mask - disp0 * mask) ** 2
    rmse = torch.sqrt(rmse.mean())
    return rmse


def AEPE(disp0, dispgt, mask=None):
    if mask is None:
        mask = torch.ones_like(dispgt).type_as(dispgt)
    aepe = torch.abs(dispgt * mask - disp0 * mask)
    aepe = aepe.mean()
    return aepe


def BadPixels(disp0, dispgt, mask=None, deltas=[1, 3, 5]):
    all_pixels_count = torch.sum(torch.ones_like(dispgt).type_as(dispgt))
    non_masked = torch.sum(mask)
    masked = all_pixels_count - non_masked
    diff = torch.abs(dispgt * mask - disp0 * mask)
    outputs = []
    for delta in deltas:
        outputs.append((torch.sum(diff > delta)) / non_masked)
    return outputs


def RMSE_log(disp0, dispgt, mask):
    rmse_log = (torch.log(disp0 + 0.000001) * mask - torch.log(dispgt + 0.000001) * mask) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    return rmse_log


def Abs_Rel(disp0, dispgt, mask):
    abs_rel = torch.mean((torch.abs(dispgt * mask - disp0 * mask) / (dispgt + 0.000001)) * mask)
    return abs_rel


def Sq_Rel(disp0, dispgt, mask):
    sq_rel = torch.mean(((dispgt * mask - disp0 * mask) ** 2 / (dispgt + 0.000001)) * mask)
    return sq_rel
