import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np
import shutil

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1  #axis=4
    m, _ = torch.max(x, dim=axis)#max in last channel
    m2, _ = torch.max(x, dim=axis, keepdim=True) #max in las channel
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def sparse_log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1  #axis=4
    m, _ = torch.max(x, dim=axis)#max in last channel
    m2, _ = torch.max(x, dim=axis, keepdim=True) #max in las channel
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def masked_log_sum_exp(mask,x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1  #axis=4
    m, _ = torch.max(x, dim=axis)#max in last channel
    m2, _ = torch.max(x, dim=axis, keepdim=True) #max in las channel
    #added part
    step1=torch.exp(x - m2)
    print("Step 1: ",step1.shape)
    step2=torch.sum(step1,dim=axis)
    print("Step 2: ", step2.shape)
    step3=m+torch.log(step2)
    print("Step 3:",step3.shape)


    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(o,x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    # x: 64 x 3 x 32 x 32; input rgb block
    # l: 64 x 100 x 32 x 32; output fro, pixelcnn; nr_mix *(1+3+3+3)

    x = x.permute(0, 2, 3,4, 1)  # 64 x 32 x 32 x 3
    l = l.permute(0, 2, 3,4, 1)  # 64 x 32 x 32 x 100
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    mask = o.permute(0, 2, 3, 4, 1)# 64 x 32 x 32 x 1 dense occupancy pattern


    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)  # 10
    logit_probs = l[:,:, :, :, :nr_mix]  # 64 x 32x 32x 10 # probabilities of choosing distributions
    # print("logit_probs",logit_probs.shape)
    l = l[:,:,:, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    # print("l",l.shape)
    means = l[:, :,:, :, :, :nr_mix]#bx64x64x64x3x10
    # print("means",means.shape)
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:,:, :, :, :, nr_mix:2 * nr_mix], min=-7.) #avoid very small scale
    # print("log_scales",log_scales.shape)
    coeffs = torch.tanh(l[:, :,:, :, :, 2 * nr_mix:3 * nr_mix])# coefficient between rgb channel
    # print("coeffs",coeffs.shape)

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False) #r bx64x64x64x3x10; moi channel trong 3 channel la cho rgb khi chua co linear
    m2 = (means[:,:, :, :, 1, :] + coeffs[:,:, :, :, 0, :]
          * x[:,:, :, :, 0, :]).view(xs[0], xs[1], xs[2], xs[3], 1, nr_mix) #g

    m3 = (means[:, :,:, :, 2, :] + coeffs[:,:, :, :, 1, :] * x[:,:, :, :, 0, :] +
          coeffs[:, :,:, :, 2, :] * x[:, :,:, :, 1, :]).view(xs[0], xs[1], xs[2], xs[3], 1, nr_mix) #b
    #print(means.shape, m2.shape, m3.shape)
    means = torch.cat((means[:, :,:, :, 0, :].unsqueeze(4), m2, m3), dim=4) #final means of rgb
    centered_x = x - means #computing cdf function
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    #log softmax first + log of distribution
    #print("Loss output before: ", log_probs.shape)
    log_probs = torch.sum(log_probs, dim=4) + log_prob_from_logits(logit_probs)

    # print("Previous LOSS: ",-torch.sum(log_sum_exp(log_probs)).detach().clone().item())

    # checking loss function

    # log_sum_exp_probs = log_sum_exp(log_probs)
    # maskf = torch.ones_like(log_sum_exp_probs).bool()
    # masked_log_sum_exp = torch.masked_select(log_sum_exp_probs, maskf)
    # print("checking sum: ",torch.sum(log_probs), torch.sum(masked_log_sum_exp))
    # print("LOSS with all 0 filtering: ",  -torch.sum(masked_log_sum_exp).detach().clone().item())

    #New loss function
    #print("Loss output after: ", log_probs.shape, logit_probs.shape)

    # print("Mask shape: ", mask.shape)
    # print("No points: ", torch.sum(mask))
    # mask = mask.expand(log_probs.size()).bool()
    # print("No points 2: ", torch.sum(mask))
    # log_probs2 = torch.masked_select(log_probs, mask) #mask =0 thi chac chan la co 1 channel ko phai =0
    # print("Loss shape after filtering: ", log_probs2.shape)
    # print("LOSS with filter: ", -torch.sum(log_sum_exp(log_probs2)).detach().clone().item())
    no_points=torch.sum(mask)
    mask = mask.squeeze(-1).bool()
    log_sum_exp_probs = log_sum_exp(log_probs)
    masked_log_sum_exp = torch.masked_select(log_sum_exp_probs,mask)

    #return torch.sum(log_sum_exp(log_probs))
    return -torch.sum(masked_log_sum_exp)/no_points


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3,4, 1)
    l = l.permute(0, 2, 3,4, 1)
    b,d,h,w,_=x.size()
    xs = [int(y) for y in x.size()]

    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :,:, :nr_mix]
    l = l[:, :, :,:, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :,:, :, :nr_mix]
    log_scales = torch.clamp(l[:, :,:, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 1.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 1.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(0.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))/(b*d*h*w)



def sparse_discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    # x: 64 x 3 x 32 x 32; input rgb block --> n x 3
    # l: 64 x 100 x 32 x 32; output fro, pixelcnn; nr_mix *(1+3+3+3) --> n x 100

    # x = x.permute(0, 2, 3,4, 1)  # 64 x 32 x 32 x 3
    # l = l.permute(0, 2, 3,4, 1)  # 64 x 32 x 32 x 100
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    #mask = o.permute(0, 2, 3, 4, 1)# 64 x 32 x 32 x 1 dense occupancy pattern


    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)  # 10
    logit_probs = l[:, :nr_mix]  # 64 x 32x 32x 10 # probabilities of choosing distributions--> n x 10
    # print("logit_probs",logit_probs.shape)
    l = l[:, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef # n x 3 x 30
    # print("l",l.shape)
    means = l[:,:,:nr_mix]#bx64x64x64x3x10 --> n x 3 x 10
    #print("means",means.shape)
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:,:, nr_mix:2 * nr_mix], min=-7.) #avoid very small scale
    # print("log_scales",log_scales.shape)
    coeffs = torch.tanh(l[:, :, 2 * nr_mix:3 * nr_mix])# coefficient between rgb channel n x 3 x 10
    # print("coeffs",coeffs.shape)

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False) #r bx64x64x64x3x10; moi channel trong 3 channel la cho rgb khi chua co linear --> n x 3 x 10
    m2 = (means[:, 1, :] + coeffs[:, 0, :]* x[:, 0, :]).view(xs[0],  1, nr_mix) #g

    m3 = (means[:,  2, :] + coeffs[:, 1, :] * x[:, 0, :] +coeffs[:,  2, :] * x[:,  1, :]).view(xs[0],  1, nr_mix) #b
    #print(means.shape, m2.shape, m3.shape)
    means = torch.cat((means[:, 0, :].unsqueeze(1), m2, m3), dim=1) #final means of rgb
    centered_x = x - means #computing cdf function
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)



    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    #log softmax first + log of distribution
    #print("Loss output before: ", log_probs.shape)
    log_probs = torch.sum(log_probs, dim=1) + log_prob_from_logits(logit_probs)


    log_sum_exp_probs = log_sum_exp(log_probs)

    return -torch.sum(log_sum_exp_probs)/(xs[0])

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 4, 1) #4x64x64x64x100
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3] #4x64x64x64x1

    # unpack parameters
    logit_probs = l[:,:, :, :, :nr_mix]
    l = l[:, :,:, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale #4x64x64x64x1 to 4x64x64x64x90x1x20

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5) # Gumbel-max trick something
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=4)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix]) # 4x64x64x64x1x10
    # select logistic parameters
    means = torch.sum(l[:, :, :,:, :, :nr_mix] * sel, dim=5) #4x64x64x64x1
    log_scales = torch.clamp(torch.sum(
        l[:, :, :,:, :, nr_mix:2 * nr_mix] * sel, dim=5), min=-7.) # #4x64x64x64x1
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u)) #sampling x
    x0 = torch.clamp(torch.clamp(x[:, :,:, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out





def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3,4, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:,:,:, :, :nr_mix]
    l = l[:,:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=4)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix]) #only pick the most probable distribution
    # select logistic parameters
    means = torch.sum(l[:,:, :, :, :, :nr_mix] * sel, dim=5)
    log_scales = torch.clamp(torch.sum(
        l[:,:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=5), min=-7.)
    coeffs = torch.sum(torch.tanh(
        l[:,:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=5)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5) #fill tensor with uniform distribution
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:,:, :, :, 0], min=-1.), max=1.)#red channel letting min and max to be -1 and 1
    x1 = torch.clamp(torch.clamp(
        x[:, :,:, :, 1] + coeffs[:, :,:, :, 0] * x0, min=-1.), max=1.) #green channel
    x2 = torch.clamp(torch.clamp(
        x[:, :,:, :, 2] + coeffs[:, :,:, :, 1] * x0 + coeffs[:, :,:, :, 2] * x1, min=-1.), max=1.)# blue channel

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=4)#concatenation rgb channels
    # put back in Pytorch ordering
    out = out.permute(0, 4, 1, 2,3) # return to pytorch form
    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def distribution_from_discretized_mix_logistic(x, l, nr_mix):
    # Pytorch ordering
    l=l.to('cpu')
    x=x.to('cpu')
    
    l = l.permute(0, 2, 3, 4, 1)#Bx64x64x64x100
    ls = [int(y) for y in l.size()]

    x = x.permute(0, 2, 3, 4, 1)  # Bx64x64x64x3
    xs = [int(y) for y in x.size()]
    #xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :, :nr_mix]
    l = l[:, :, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])#bx64x64x64x3x30
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    #temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data #- torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=4)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # only pick the most probable distribution
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :, :nr_mix] * sel, dim=5) #bx64x64x64x3
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, :, nr_mix:2 * nr_mix] * sel, dim=5), min=-7.)#bx64x64x64x3
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=5)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x_r = torch.clamp(torch.clamp(x[:, :, :, :, 0], min=-1.), max=1.)
    x_g = torch.clamp(torch.clamp(x[:, :, :, :, 1], min=-1.), max=1.)
    x_b = torch.clamp(torch.clamp(x[:, :, :, :, 2], min=-1.), max=1.)
    mean_r = means[:, :, :, :, 0]
    mean_g = means[:, :, :, :, 1] + coeffs[:, :, :, :, 0] * x_r# green channel
    mean_b = means[:, :, :, :, 2] + coeffs[:, :, :,:, 1] * x_r + coeffs[:, :, :,:, 2] * x_g
    means = torch.cat([mean_r.view(xs[:-1] + [1]), mean_g.view(xs[:-1] + [1]), mean_b.view(xs[:-1] + [1])],
                    dim=4)  # concatenation mean of rgb channels bx64x64x64x3

    print('means:', means.shape, means.min(), means.mean(), means.max())
    #print("Mean and log scale output, x input ",means.shape, log_scales.shape, x.shape)
    # log_scales_r=log_scales[]
    # log_scales_g=log_scales_r+coeffs[:, :, :, :, 0]*coeffs[:, :, :, :, 0]*log_scales_r
    # log_scales_b=



    sample_x=torch.FloatTensor(torch.Size(xs[:-1]+[3,254]))#bx64x64x64x3x254
    for idx in range(254):
        sample_x[:,:,:,:,:,idx]=((idx+1)-127.5)/127.5
    means_expanded=means.unsqueeze(-1).repeat(1,1,1,1,1,254)
    log_scales_expanded = log_scales.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 254)
    #print("changing shape: ",sample_x.shape,means.shape, log_scales.shape, means_expanded[0,0,0,0,0,0:3],log_scales_expanded[0,0,0,0,0,0:3],sample_x[0,0,0,0,0,0:3])

    #sum_pdf=0
    if means.is_cuda: sample_x = sample_x.cuda()
    #for idx in range(254):
        #sample_x=(torch.ones(x.size())*(idx+1)-127.5)/127.5

    centered_x=sample_x-means_expanded
    inv_stdv = torch.exp(-log_scales_expanded)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    cdf_delta=cdf_plus-cdf_min
    #print("cdf delta: ",torch.min(cdf_delta), torch.max(cdf_delta), torch.sum(cdf_delta))
    #sum_pdf+=cdf_delta[0,1,2,3,0].detach().cpu().item()


    #at 0 and 255, special cases
    sample_black=(torch.zeros(x.size())-127.5)/127.5
    if means.is_cuda: sample_black = sample_black.cuda()
    centered_x=sample_black-means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    black_pixel = torch.exp(plus_in - F.softplus(plus_in)) #bx64x64x64x3
    #print("Black pixel: ", black_pixel.shape, black_pixel.min(), black_pixel.max())

    sample_white = (torch.ones(x.size()) * 255. - 127.5) / 127.5
    if means.is_cuda: sample_white = sample_white.cuda()
    centered_x = sample_white - means
    inv_stdv = torch.exp(-log_scales)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    white_pixel = torch.exp(-F.softplus(min_in))
    #print("White pixel: ", white_pixel.shape, white_pixel.min(),white_pixel.max())

    out=torch.cat([black_pixel.view(xs+[1]),cdf_delta,white_pixel.view(xs+[1])],dim=5)
    #print('output shape: ', out.shape, out.sum(dim=5).min(),out.sum(dim=5).mean(),out.sum(dim=5).max())
    #out = cdf_delta


    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    # mid_in = inv_stdv * centered_x
    # log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)


    #print("Test means and output softmax: ",means[0, 0, 0, 0, 0:5],log_scales[0, 0, 0, 0, 0:3], torch.argmax(out, dim=5)[0,0,0,0,0:5])
    out=out.permute(0,4,5,1,2,3)
    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def distribution_from_discretized_mix_logistic2(x, l, nr_mix):
    # Pytorch ordering
    l=l.to('cpu')
    x=x.to('cpu')
    
    l = l.permute(0, 2, 3, 4, 1)#Bx64x64x64x100
    ls = [int(y) for y in l.size()]

    x = x.permute(0, 2, 3, 4, 1)  # Bx64x64x64x3
    xs = [int(y) for y in x.size()]
    #xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :, :nr_mix]
    l = l[:, :, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])#bx64x64x64x3x30
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    #temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data #- torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=4)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # only pick the most probable distribution
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :, :nr_mix] * sel, dim=5) #bx64x64x64x3
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, :, nr_mix:2 * nr_mix] * sel, dim=5), min=-7.)#bx64x64x64x3
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=5)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x_r = torch.clamp(torch.clamp(x[:, :, :, :, 0], min=-1.), max=1.)
    x_g = torch.clamp(torch.clamp(x[:, :, :, :, 1], min=-1.), max=1.)
    x_b = torch.clamp(torch.clamp(x[:, :, :, :, 2], min=-1.), max=1.)
    mean_r = means[:, :, :, :, 0]
    mean_g = means[:, :, :, :, 1] + coeffs[:, :, :, :, 0] * x_r# green channel
    mean_b = means[:, :, :, :, 2] + coeffs[:, :, :,:, 1] * x_r + coeffs[:, :, :,:, 2] * x_g
    means = torch.cat([mean_r.view(xs[:-1] + [1]), mean_g.view(xs[:-1] + [1]), mean_b.view(xs[:-1] + [1])],
                    dim=4)  # concatenation mean of rgb channels bx64x64x64x3

    #print('means:', means.shape, means.min(), means.mean(), means.max())



    sample_x=torch.FloatTensor(torch.Size(xs[:-1]+[3,256]))#bx64x64x64x3x254
    for idx in range(256):
        sample_x[:,:,:,:,:,idx]=(idx-127.5)/127.5
    means_expanded=means.unsqueeze(-1).repeat(1,1,1,1,1,256)
    log_scales_expanded = log_scales.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 256)

    if means.is_cuda: sample_x = sample_x.cuda()


    centered_x=sample_x-means_expanded
    inv_stdv = torch.exp(-log_scales_expanded)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales_expanded - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (sample_x > 0.9999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (sample_x < -0.9999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    out=torch.exp(log_probs)
    #print('output shape: ', out.min(), out.max(), out.mean())
    #print("Test means and output softmax: ",means[0, 0, 0, 0, 0:5],log_scales[0, 0, 0, 0, 0:3], torch.argmax(out, dim=5)[0,0,0,0,0:5])
    out=out.permute(0,4,5,1,2,3)
    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''


def sparse_distribution_from_discretized_mix_logistic2(x, l, nr_mix):
    # Pytorch ordering
    # l = l.to('cpu')
    # x = x.to('cpu')

    # l = l.permute(0, 2, 3, 4, 1)  # Bx64x64x64x100 -->nx100
    ls = [int(y) for y in l.size()]

    # x = x.permute(0, 2, 3, 4, 1)  # Bx64x64x64x3 --> nx3
    xs = [int(y) for y in x.size()]
    # xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :nr_mix]
    l = l[:,  nr_mix:].contiguous().view(xs + [nr_mix * 3])  # bx64x64x64x3x30-->nx3x30
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    # temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data  # - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=1)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # only pick the most probable distribution
    # select logistic parameters
    means = torch.sum(l[:,:, :nr_mix] * sel, dim=2)  # bx64x64x64x3
    log_scales = torch.clamp(torch.sum(
        l[:, :,  nr_mix:2 * nr_mix] * sel, dim=2), min=-7.)  # bx64x64x64x3
    coeffs = torch.sum(torch.tanh(
        l[:, :,  2 * nr_mix:3 * nr_mix]) * sel, dim=2)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x_r = torch.clamp(torch.clamp(x[:,  0], min=-1.), max=1.)
    x_g = torch.clamp(torch.clamp(x[:,  1], min=-1.), max=1.)
    x_b = torch.clamp(torch.clamp(x[:,  2], min=-1.), max=1.)
    mean_r = means[:,  0]
    mean_g = means[:,  1] + coeffs[:,  0] * x_r  # green channel
    mean_b = means[:,   2] + coeffs[:,  1] * x_r + coeffs[:, 2] * x_g
    means = torch.cat([mean_r.view(xs[:-1] + [1]), mean_g.view(xs[:-1] + [1]), mean_b.view(xs[:-1] + [1])],
                      dim=1)  # concatenation mean of rgb channels bx64x64x64x3

    # print('means:', means.shape, means.min(), means.mean(), means.max())

    sample_x = torch.FloatTensor(torch.Size(xs[:-1] + [3, 256]))  # bx64x64x64x3x254
    for idx in range(256):
        sample_x[:,  :, idx] = (idx - 127.5) / 127.5
    means_expanded = means.unsqueeze(-1).repeat(1, 1, 256)
    log_scales_expanded = log_scales.unsqueeze(-1).repeat(1,  1, 256)

    if means.is_cuda: sample_x = sample_x.cuda()

    centered_x = sample_x - means_expanded
    inv_stdv = torch.exp(-log_scales_expanded)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales_expanded - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (sample_x > 0.9999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (sample_x < -0.9999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    out = torch.exp(log_probs) #
    #print('output shape: ',out.shape, out.min(), out.max(), out.mean())
    # print("Test means and output softmax: ",means[0, 0, 0, 0, 0:5],log_scales[0, 0, 0, 0, 0:3], torch.argmax(out, dim=5)[0,0,0,0,0:5])
    # out = out.permute(0,2,1)
    return out


def sparse_narrow_distribution_from_discretized_mix_logistic2(x, l, nr_mix):
    # Pytorch ordering
    # l = l.to('cpu')
    # x = x.to('cpu')

    # l = l.permute(0, 2, 3, 4, 1)  # Bx64x64x64x100 -->nx100
    ls = [int(y) for y in l.size()]

    # x = x.permute(0, 2, 3, 4, 1)  # Bx64x64x64x3 --> nx3
    xs = [int(y) for y in x.size()]
    # xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :nr_mix]
    l = l[:,  nr_mix:].contiguous().view(xs + [nr_mix * 3])  # bx64x64x64x3x30-->nx3x30
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    # temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data  # - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=1)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # only pick the most probable distribution
    # select logistic parameters
    means = torch.sum(l[:,:, :nr_mix] * sel, dim=2)  # bx64x64x64x3
    log_scales = torch.clamp(torch.sum(
        l[:, :,  nr_mix:2 * nr_mix] * sel, dim=2), min=-7.)  # bx64x64x64x3
    coeffs = torch.sum(torch.tanh(
        l[:, :,  2 * nr_mix:3 * nr_mix]) * sel, dim=2)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x_r = torch.clamp(torch.clamp(x[:,  0], min=-1.), max=1.)
    x_g = torch.clamp(torch.clamp(x[:,  1], min=-1.), max=1.)
    x_b = torch.clamp(torch.clamp(x[:,  2], min=-1.), max=1.)
    mean_r = means[:,  0]
    mean_g = means[:,  1] + coeffs[:,  0] * x_r  # green channel
    mean_b = means[:,   2] + coeffs[:,  1] * x_r + coeffs[:, 2] * x_g
    means = torch.cat([mean_r.view(xs[:-1] + [1]), mean_g.view(xs[:-1] + [1]), mean_b.view(xs[:-1] + [1])],
                      dim=1)  # concatenation mean of rgb channels nx3

    # print('means:', means.shape, means.min(), means.mean(), means.max())

    sample_x = torch.FloatTensor(torch.Size(xs[:-1] + [3, 256]))  # bx64x64x64x3x254
    for idx in range(256):
        sample_x[:,  :, idx] = (idx - 127.5) / 127.5
    means_expanded = means.unsqueeze(-1).repeat(1, 1, 256)
    log_scales_expanded = log_scales.unsqueeze(-1).repeat(1,  1, 256) #nx3x256
    log_scales_expanded = log_scales_expanded-0.08*log_scales_expanded

    if means.is_cuda: sample_x = sample_x.cuda()

    centered_x = sample_x - means_expanded
    inv_stdv = torch.exp(-log_scales_expanded)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases #probability are not in -1, 1
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales_expanded - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (sample_x > 0.9999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (sample_x < -0.9999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    out = torch.exp(log_probs) #
    #out=out/torch.max(torch.sum(out,dim=2))
    #print('output shape: ',out.shape, out.min(), out.max(), out.mean())
    # print("Test means and output softmax: ",means[0, 0, 0, 0, 0:5],log_scales[0, 0, 0, 0, 0:3], torch.argmax(out, dim=5)[0,0,0,0,0:5])
    # out = out.permute(0,2,1)
    top10, i10 = torch.topk(out, 10, 2, True, True)
    #print('Index shape: ',i10.shape)
    #torch.set_printoptions(precision=10)
    sum_top10=torch.sum(top10,dim=2)
    #print('sumtop shape: ', sum_top10.shape, torch.min(sum_top10[:]), torch.max(sum_top10[:]), torch.sum(sum_top10[:]))
    res=(1.0-sum_top10)/10.0
    #print('res shape: ', res.shape, torch.min(res[:]), torch.max(res[:]), torch.sum(res[:]))
    res=res.unsqueeze(-1).repeat(1, 1, 10)

    refined_out=torch.zeros_like(out,device=out.device)
    refined_out[:]=1e-9/245

    ind = torch.from_numpy(np.indices(i10.shape))
    ind[-1] = i10
    refined_out[tuple(ind)]=out[tuple(ind)]+res-1e-10
    refined_out=refined_out.clamp(min=1e-10)
    #if(not torch.all(refined_out>=0)):
    #    print(torch.min(refined_out[:]),torch.max(refined_out[:]), torch.sum(refined_out[:]))


    return refined_out

def distribution_from_mix_logistic_loss_1d(l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering

    l = l.permute(0, 2, 3,4, 1)
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :,:, :nr_mix]
    l = l[:, :, :,:, nr_mix:].contiguous().view(ls[:-1]+[1] + [nr_mix * 2])  # 2 for mean, scale
    # means = l[:, :, :,:,:, :nr_mix]
    # log_scales = torch.clamp(l[:, :,:, :,:, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding

    # sample mixture indicator from softmax

    #temp = torch.FloatTensor(logit_probs.size())
    #if l.is_cuda: temp = temp.cuda()
    #temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data# - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=4)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(ls[:-1] + [1, nr_mix])  # only pick the most probable distribution


    # select logistic parameters
    #print("selection: ",nr_mix, l.shape, means.shape, sel.shape)
    means = torch.sum(l[:, :, :, :,:, :nr_mix] * sel, dim=5)  # bx64x64x64x3
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, :, nr_mix:2 * nr_mix] * sel, dim=5), min=-7.)  # bx64x64x64x3
    #print("selection: ", nr_mix, l.shape,sel.shape, means.shape, log_scales.shape)
    # sub-pixels
    # print('Input shape 1: ', x.shape)
    sample_x = torch.FloatTensor(torch.Size(ls[:-1] + [1])).to(means.device)  # bx64x64x64x2
    for idx in range(1):
        sample_x[:, :, :, :, idx] = 2*idx-1
    means_expanded = means.repeat(1, 1, 1, 1, 1)
    log_scales_expanded = log_scales.repeat(1, 1, 1, 1, 1)
    # x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    #print('Input shape 2: ', sample_x.shape, means_expanded.shape, log_scales_expanded.shape)


    centered_x = sample_x - means_expanded
    inv_stdv = torch.exp(-log_scales_expanded)
    #plus_in = inv_stdv * (centered_x + 1. / 1.)
    # cdf_plus = torch.sigmoid(plus_in)
    # min_in = inv_stdv * (centered_x - 1. / 1.)
    # cdf_min = torch.sigmoid(min_in)
    #cdf_delta = cdf_plus - cdf_min  # probability for all other cases torch.Size([1, 64, 64, 64, 1]
    #print('cdf delta shape: ', cdf_delta.shape)

    plus_in = inv_stdv * (centered_x + 1. / 1.)
    black_pixel = torch.exp(plus_in - F.softplus(plus_in)) # [1, 64, 64, 64, 1]
    white_pixel=1-black_pixel
    out=torch.cat((black_pixel,white_pixel),dim=4)
    out = out.permute(0, 4, 1, 2, 3)
    return out


def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))


def compute_metric(predict, target,writer, step):
    pred_label = torch.argmax(predict, dim=1)
    tp = torch.count_nonzero(pred_label * target)
    fp = torch.count_nonzero(pred_label * (target - 1))
    tn = torch.count_nonzero((pred_label - 1) * (target - 1))
    fn = torch.count_nonzero((pred_label - 1) * (target))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    writer.add_scalar("bc/precision", precision,step)
    writer.add_scalar("bc/recall", recall,step)
    writer.add_scalar("bc/accuracy", accuracy,step)
    writer.add_scalar("bc/specificity", specificity,step)
    writer.add_scalar("bc/f1_score", f1,step)
    return tp.item(), fp.item(), tn.item(), fn.item(), precision.item(), recall.item(), accuracy.item(), specificity.item(), f1.item()
def compute_metric_MoL(predict, target,writer, step):
    pred_label = predict
    tp = torch.count_nonzero(pred_label * target)
    fp = torch.count_nonzero(pred_label * (target - 1))
    tn = torch.count_nonzero((pred_label - 1) * (target - 1))
    fn = torch.count_nonzero((pred_label - 1) * (target))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    writer.add_scalar("bc/precision", precision,step)
    writer.add_scalar("bc/recall", recall,step)
    writer.add_scalar("bc/accuracy", accuracy,step)
    writer.add_scalar("bc/specificity", specificity,step)
    writer.add_scalar("bc/f1_score", f1,step)
    return tp.item(), fp.item(), tn.item(), fn.item(), precision.item(), recall.item(), accuracy.item(), specificity.item(), f1.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min
def load_ckp_not_optimizer(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer

    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, checkpoint['epoch'], valid_loss_min

def greedy_load_ckp(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """

    checkpoint = torch.load(checkpoint_fpath)

    pretrained_dict=checkpoint['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model

def save_entire_ckp(model,state, is_best,model_path, checkpoint_path, best_model_path):
    """
    model: network architecture
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    #save the entire model
    torch.save(model,model_path)
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_entire_ckp(model_path,checkpoint_fpath, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    #load the net architecture
    model=torch.load(model_path)
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min

def load_part_of_model(model, path):

    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model
