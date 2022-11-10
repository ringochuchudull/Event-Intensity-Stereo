import torch
import numpy as np

import torchvision.transforms.functional as tF

from PIL import Image
import cv2

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


def parsing_event_txt_line(line):
    time, x, y, p = line.strip().split(' ')
    return [float(time), float(x), float(y), float(p)]


def events2bin(events, max_w, max_h):
    event_bin = torch.zeros((2, max_h, max_w))
    start_time = events[0][0]; end_time = events[-1][0]
    time_interval = end_time - start_time
    for event in events:
        time, w, h, p = event
        p_index = p
        event_bin[p_index, h, w] += 1
    return event_bin, torch.arange(1) + time_interval


def events2bins(events, max_w, max_h, num_bins):
    """
    :param events: a batch of events [[time, x, y, p], [], ...,]] size (N, 4), N is the number of events
    :param max_w: max x index
    :param max_h: max h index
    :param num_bins:
    :return: event_bins (2, num_bins, max_h, max_w)
    """
    event_bins = torch.zeros((2, num_bins, max_h, max_w))
    start_time = events[0][0]; end_time = events[-1][0]
    time_interval = (end_time - start_time) / num_bins
    for event in events:
        time, w, h, p = event
        p_index = p
        bin_index = (time - start_time) // time_interval
        if bin_index == num_bins:
            bin_index = num_bins - 1
        event_bins[int(p_index), int(bin_index), int(h), int(w)] += 1
    rtime_of_bins = torch.arange(num_bins) * time_interval
    return event_bins, rtime_of_bins


def events_merge_bins(event_bins, new_num_bins):
    """
    :param event_bins: (2, num_bins, max_h, max_w) or (B, 2, num_bins, max_h, max_w)
    :param new_num_bins:
    :return:
    """
    if event_bins.ndim == 4:
        _, num_bins, max_h, max_w = event_bins.shape
        return torch.sum(event_bins.view((2, int(num_bins / new_num_bins), new_num_bins, max_h, max_w)), 1)
    elif event_bins.ndim == 5:
        B, _, num_bins, max_h, max_w = event_bins.shape
        return torch.sum(event_bins.view((B, 2, int(num_bins / new_num_bins), new_num_bins, max_h, max_w)), 2)
    else:
        raise Exception(f'Wrong size of event bins, expect (2, num_bins, max_h, max_w) or (B, 2, num_bins, max_h, max_w), received {event_bins.shape}')


def find_events_in_time(events, r_time=0.01):
    index = 0
    start_time = events[0][0]
    for i, event in enumerate(events):
        if event[0] > (start_time + r_time):
            index += i
            break
    return events[:index]


def plot_3d_event(events, max_h=180, max_w=240, view=(5, 60)):
    es = np.array(events)
    es_p = es[es[:, 3] == 1]
    es_n = es[es[:, 3] == 0]
    tp = es_p[:, 0]; xp = es_p[:, 1]; yp = es_p[:, 2]
    tn = es_n[:, 0]; xn = es_n[:, 1]; yn = es_n[:, 2]

    fig = plt.figure(figsize=(10,8))
    ax = Axes3D(fig)
    ax.scatter(max_w - xp, tp, max_h - yp, c='blue', s=1)
    ax.scatter(max_w - xn, tn, max_h - yn, c='red', s=1)
    ax.view_init(view[0], view[1])
    ax.set_zlabel('y', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('time', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('x', fontdict={'size': 15, 'color': 'black'})
    plt.show()

def plot_2d_event_bin(event_bin):
    """
    :param event_bin:  (2, H, W)
    :return:
    """
    if event_bin.requires_grad:
        event_bin_np = event_bin.detach().numpy()
    else:
        event_bin_np = event_bin.numpy()
    cmap_0 = cm.get_cmap("seismic_r")
    cmap_1 = cm.get_cmap("seismic")
    norm_0 = colors.Normalize()
    norm_1 = colors.Normalize()

    norm_0.autoscale(event_bin_np[0])
    norm_1.autoscale(event_bin_np[1])

    normalised_0 = norm_0(event_bin_np[0]) * 0.5 + 0.5
    normalised_1 = norm_1(event_bin_np[1]) * 0.5 + 0.5
    norm_0_pil = Image.fromarray(cv2.cvtColor(cmap_0(normalised_0, bytes=True), cv2.COLOR_BGRA2RGB))
    norm_1_pil = Image.fromarray(cv2.cvtColor(cmap_1(normalised_1, bytes=True), cv2.COLOR_BGRA2RGB))
    return Image.blend(norm_0_pil, norm_1_pil, 0.5)

def plot_2d_events(events, max_w, max_h):
    event_bin = events2bin(events, max_w, max_h)[0]
    eb_r = event_bin[1:, :, :].clone()
    eb_b = event_bin[:1, :, :].clone()
    eb_g = torch.ones_like(eb_r)
    index = (eb_r + eb_b) > 0
    eb_g[index] = 0
    eb_r[~index] = 1
    eb_b[~index] = 1
    eb_addg = torch.cat([eb_r, eb_g, eb_b], 0)
    return tF.to_pil_image(eb_addg)


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return tF.to_pil_image(np.uint8(img))


def gen_flow_circle(center, height, width):
    x0, y0 = center
    if x0 >= height or y0 >= width:
        raise AttributeError('ERROR')
    flow = np.zeros((height, width, 2), dtype=np.float32)

    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

    grid_x0 = np.tile(np.array([x0]), [height, width])
    grid_y0 = np.tile(np.array([y0]), [height, width])

    flow[:,:,0] = grid_x0 - grid_x
    flow[:,:,1] = grid_y0 - grid_y
    return flow
