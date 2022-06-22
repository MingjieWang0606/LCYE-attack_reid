from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from util.spectral import SpectralNorm
from util.gumbel import gumbel_softmax
import numpy as np
import math
from memory_module import MemModule

class CollaspeNet(nn.Module):
  """
  Defines a Coallspe Nemory Net
  """
  def __init__(self, num_classes, pool_dim, mem_dim, n_upsampling, temperature, use_gumbel,):
      super(CollaspeNet,self).__init__()
      self.temperature = temperature
      self.use_gumbel = use_gumbel
      self.pool_dim = pool_dim
      self.n_upsampling = n_upsampling
      self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=self.pool_dim)
      ## reid branch
      self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = self._construct_fc_layer(
          [self.pool_dim], self.pool_dim, dropout_p=None
      )
      self.classifier = nn.Linear(self.pool_dim, num_classes)
      ## mask branch
      self.up_layers = nn.ModuleList()
      self.gmp = nn.AdaptiveMaxPool2d((1,1))#or average pool
      # upsample
      for i in range(self.n_upsampling):
        
        self.up_layers.append(nn.Sequential(SpectralNorm(nn.ConvTranspose2d(pool_dim, int(pool_dim/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)), nn.LeakyReLU(0.2, True)))
        pool_dim = int(pool_dim/2)
      #self.up_layers = nn.Sequential(*up_layers)

      self.end_1 = nn.Sequential(*[nn.Conv2d(pool_dim, 1, kernel_size=1, stride=1)])
      self.logsoftmax = nn.LogSoftmax(dim=1)


      self._init_params()

  def forward(self,spatial_feat, target_label=None, G_or_D = 'G'):
    if G_or_D == 'C':
      # reid flow
      mem, mem_att = self.mem_rep(spatial_feat, mode='w')
      x = spatial_feat + mem
      x_pool = self.global_avgpool(x)
      x_pool = x_pool.view(x_pool.size(0), -1)
      x_pool = self.fc(x_pool) # b 2048
      x_feat = self.classifier(x_pool)

      # mask flow
      out = self.gmp(spatial_feat)*spatial_feat
      for i in range(self.n_upsampling):
        out = self.up_layers[i](out)
      
      out = self.end_1(out)

      n,c,h,w = out.size()
      if self.temperature == -1: return x_feat, x_pool, torch.ones((n,1,h,w)).cuda()

      if not self.use_gumbel:
        logits = self.logsoftmax(out.view(n, -1))
        th, _ = torch.topk(logits, k=int(self.temperature), dim=1, largest=True)
        mask, zeros, ones = torch.zeros_like(logits).cuda(), torch.zeros(h*w).cuda(), torch.ones(h*w).cuda()
        for i in range(n):
          mask[i,:] = torch.where(logits[i,:]>=th[i, int(self.temperature)-1], ones, zeros)
        mask = mask.view(n, 1, h, w)
      elif self.use_gumbel:
        logits = gumbel_softmax(out.view(n, -1), k=int(self.temperature), T=self.T, hard=True, eps=1e-10).view(n, 1, h, w)
        mask = logits.cuda()

      return x_feat, x_pool, mask

    elif G_or_D == 'D':
      x_mem, _ = self.mem_rep(spatial_feat, mode = 'Dr')
      return x_mem + spatial_feat
    elif G_or_D == 'G':
      x_mem,_ = self.mem_rep(spatial_feat, target_label, mode='Gr')
      return x_mem + spatial_feat
    else:
      raise ValueError('Wrong flow selection')


  def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
      """Constructs fully connected layer

      Args:
          fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
          input_dim (int): input dimension
          dropout_p (float): dropout probability, if None, dropout is unused
      """
      if fc_dims is None:
          self.feature_dim = input_dim
          return None

      assert isinstance(
          fc_dims, (list, tuple)
      ), 'fc_dims must be either list or tuple, but got {}'.format(
          type(fc_dims)
      )

      layers = []
      for dim in fc_dims:
          layers.append(nn.Linear(input_dim, dim))
          layers.append(nn.BatchNorm1d(dim))
          layers.append(nn.ReLU(inplace=True))
          if dropout_p is not None:
              layers.append(nn.Dropout(p=dropout_p))
          input_dim = dim

      self.feature_dim = fc_dims[-1]

      return nn.Sequential(*layers)

  def _init_params(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(
                  m.weight, mode='fan_out', nonlinearity='relu'
              )
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm1d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)



class Pat_Discriminator(nn.Module):
  """
  Defines a PatchGAN discriminator
  Code based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
  """
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='bn'):
    """Construct a PatchGAN discriminator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
    super(Pat_Discriminator, self).__init__()

    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
      use_bias = norm_layer.func != nn.BatchNorm2d
    else:
      use_bias = norm_layer != nn.BatchNorm2d

    kw = 4
    padw = 1
    sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):  # gradually increase the number of filters
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
    self.model = nn.Sequential(*sequence)

  def forward(self, x):
    return self.model(x), torch.ones_like(x)


class MS_Discriminator(nn.Module):
  def __init__(self, input_nc, num_classes, ndf=64, n_layers=3, norm='bn', num_D=3, pool_dim =2048):
    super(MS_Discriminator, self).__init__()
    self.num_D = num_D
    self.n_layers = n_layers
    self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, count_include_pad=False)
    self.same0 = SamePadding(kernel_size=3, stride=2)
    self.same1 = SamePadding(kernel_size=4, stride=2)
    self.same2 = SamePadding(kernel_size=4, stride=1)
    #self.Mask = Mask(norm, temperature, use_gumbel)

    for i in range(num_D):
      netD = sub_Discriminator(input_nc, num_classes, ndf, n_layers, norm)
      channel_in = nn.Conv2d(512, pool_dim,kernel_size = 1, stride=1)
      channel_out = nn.Conv2d(pool_dim, 512,kernel_size = 1, stride=1)
      setattr(self, 'D'+str(i)+'_channel_in', channel_in)
      setattr(self, 'D'+str(i)+'_channel_out', channel_out)
      for j in range(n_layers+2): setattr(self, 'D'+str(i)+'_layer'+str(j), getattr(netD, 'layer'+str(j)))                                   

  def single_forward(self, model, x, idx,C =None):
    result = [x]
    for i in range(len(model)): #j
      samepadding = self.same1 if i < len(model)-2 else self.same2
      out = model[i](samepadding(result[-1]))
      if out.size(1) == 512 and C !=None and idx == 1:
        out_in = getattr(self, 'D'+str(idx)+'_channel_in')(out)
        out_in = C(out_in, G_or_D = 'D') + out_in
        out = getattr(self, 'D'+str(idx)+'_channel_out')(out_in)

      result.append(out)
    return result[1:]

  def forward(self, x, C=None):        
    num_D = self.num_D
    proposal = []
    result = []
    mask = None
    input_downsampled = x
    for i in range(num_D):
      model = [getattr(self, 'D'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
      proposal.append(self.single_forward(model, input_downsampled, i, C)) #[[D2L0, D2L1,..., D2L4],[D1L0,...,D1L4],[D0L0,...,D0L4]]
      if i != (num_D-1): input_downsampled = self.downsample(self.same0(input_downsampled))
    for i in proposal: result.append(i[-1])
    #mask = self.Mask(x, proposal)
    return result, mask
        
# (64,128,256,512,1) 
class sub_Discriminator(nn.Module):
  def __init__(self, input_nc, num_classes, ndf=64, n_layers=3, norm='in'):
    super(sub_Discriminator, self).__init__()
    self.n_layers = n_layers

    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    sequence = [[SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, bias=use_bias)), nn.LeakyReLU(0.2, True)]]
    nf = ndf
    for n in range(1, n_layers):
      nf_prev = nf
      nf = min(nf*2, 512)
      sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]

    nf_prev = nf
    nf = min(nf*2, 512)
    sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
    sequence += [[nn.Conv2d(nf, num_classes+1, kernel_size=4, stride=1)]]

    for n in range(len(sequence)):
      setattr(self, 'layer'+str(n), nn.Sequential(*sequence[n]))

  def forward(self, input):
    res = [input]
    for n in range(self.n_layers+2):
      model = getattr(self, 'layer'+str(n))
      res.append(model(res[-1]))
    return res[1:]

class Mask(nn.Module):
  def __init__(self, norm, temperature, use_gumbel, fused=1):
    super(Mask, self).__init__()
    self.temperature = temperature
    self.use_gumbel = use_gumbel
    self.fused = fused
    self.T = nn.Parameter(torch.Tensor([1]))
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    small_channels = [512, 512, 256, 128]
    big_channels = [512+128, 512+128+64, 128+64, 64] if self.fused == 2 else [512, 512, 128, 64]

    self.up32_16 = UpLayer(big_channels=big_channels[0], out_channels=512, small_channels=small_channels[0], norm_layer=norm_layer)
    self.up16_8 = UpLayer(big_channels=big_channels[1], out_channels=256, small_channels=small_channels[1], norm_layer=norm_layer)
    self.up8_4 = UpLayer(big_channels=big_channels[2], out_channels=128, small_channels=small_channels[2], norm_layer=norm_layer)
    # self.up4_2 = UpLayer(big_channels=big_channels[3], out_channels=64, small_channels=small_channels[3], norm_layer=norm_layer)
    self.deconv1 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.deconv2 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[nn.Conv2d(128, 1, kernel_size=1, stride=1)])
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, x, proposal):
    n,c,h,w = x.size()
    if self.temperature == -1: return torch.ones((n,1,h,w)).cuda()
    scale32 = proposal[2][3]
    scale16 = torch.cat((proposal[2][1], proposal[1][3]),1) if self.fused == 2 else proposal[1][3]
    scale8 = torch.cat((proposal[0][3], proposal[1][1], proposal[2][0]),1) if self.fused == 2 else proposal[0][3]
    scale4 = torch.cat((proposal[0][1], proposal[1][0]),1) if self.fused == 2 else proposal[0][1]
    scale2 = proposal[0][0]
    out = self.up32_16(scale32, scale16)
    out = self.up16_8(out, scale8)
    out = self.up8_4(out, scale4)
    # out = self.up4_2(out, scale2)
    out = self.deconv1(out) 
    out = self.deconv2(out) 
    out = self.conv2(out)

    if not self.use_gumbel:
      logits = self.logsoftmax(out.view(n, -1))
      th, _ = torch.topk(logits, k=int(self.temperature), dim=1, largest=True)
      mask, zeros, ones = torch.zeros_like(logits).cuda(), torch.zeros(h*w).cuda(), torch.ones(h*w).cuda()
      for i in range(n):
        mask[i,:] = torch.where(logits[i,:]>=th[i, int(self.temperature)-1], ones, zeros)
      mask = mask.view(n, 1, h, w)
    elif self.use_gumbel:
      logits = gumbel_softmax(out.view(n, -1), k=int(self.temperature), T=self.T, hard=True, eps=1e-10).view(n, 1, h, w)
      mask = logits.cuda()
      # logits = F.gumbel_softmax(out.view(n, -1), tau=self.temperature).view(n, 1, h, w)
      # # logits_normed = torch.clamp((logits_normed+1e-4), min=0, max=1)
      # logits = np.minimum(1.0, logits.data.cpu().numpy()*(h*w)+1e-4)
      # mask = torch.bernoulli(torch.from_numpy(logits)).cuda()
    return mask

class UpLayer(nn.Module):
  def __init__(self, big_channels, out_channels, small_channels, norm_layer):
    super(UpLayer, self).__init__()
    self.big_channels = big_channels
    self.out_channels = out_channels
    self.small_channels = small_channels
    self.conv1 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.big_channels, self.small_channels, kernel_size=1, stride=1)), norm_layer(self.small_channels), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.small_channels, self.out_channels, kernel_size=3, stride=1, padding=1)), norm_layer(self.out_channels), nn.LeakyReLU(0.2, True)])

  def forward(self, small, big):
    small = F.upsample(small, size=(big.size()[2], big.size()[3]), mode='bilinear')
    big = self.conv1(big)
    out = self.conv2(big+small)
    return out

class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', pool_dim=2048, n_blocks=6):
    super(Generator, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)

    self.channel_in_G = nn.Conv2d(128, pool_dim, kernel_size=1, stride=1)
    self.channel_out_G = nn.Conv2d(pool_dim, 128, kernel_size=1, stride=1)

  def forward_encode(self, inputs):
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)
    out = self.channel_in_G(out)
    return out

  def forward_decode(self, out):
    out = self.channel_out_G(out)
    out = self.l4_1(out)
    out = self.l4_2(out)
    out = self.l5(out)
    return out

  def forward(self, input, En_or_De = 'E'):
    if En_or_De == 'E':
      return self.forward_encode(input)
    elif En_or_De == 'D':
      return self.forward_decode(input)
    else:
      raise ValueError('Unknown flow')

class ResnetG(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=6):
    super(ResnetG, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    begin_layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)

  def forward(self, inputs):
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)
    out = self.l4_1(out)
    out = self.l4_2(out)
    out = self.l5(out)
    return out

# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, use_bias):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

  def build_conv_block(self, dim, norm_layer, use_bias):
    conv_block = []
    for i in range(2):
      conv_block += [nn.ReflectionPad2d(1)]
      conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)), norm_layer(dim)]
      if i < 1: 
        conv_block += [nn.ReLU(True)]
    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

class SamePadding(nn.Module):
  def __init__(self, kernel_size, stride):
    super(SamePadding, self).__init__()
    self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
    self.stride = torch.nn.modules.utils._pair(stride)

  def forward(self, input):
    in_width = input.size()[2]
    in_height = input.size()[3]
    out_width = math.ceil(float(in_width) / float(self.stride[0]))
    out_height = math.ceil(float(in_height) / float(self.stride[1]))
    pad_along_width = ((out_width - 1) * self.stride[0] +
                       self.kernel_size[0] - in_width)
    pad_along_height = ((out_height - 1) * self.stride[1] +
                        self.kernel_size[1] - in_height)
    pad_left = int(pad_along_width / 2)
    pad_top = int(pad_along_height / 2)
    pad_right = pad_along_width - pad_left
    pad_bottom = pad_along_height - pad_top
    return F.pad(input, (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)), 'constant', 0)

  def __repr__(self):
    return self.__class__.__name__

def weights_init(m):
  classname = m.__class__.__name__
  # print(dir(m))
  if classname.find('Conv') != -1:
    if 'weight' in dir(m): 
      m.weight.data.normal_(0.0, 1)
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

class GANLoss_real_fake(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.cuda.FloatTensor):
    super(GANLoss_real_fake, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_var = None
    self.fake_label_var = None
    self.Tensor = tensor
    if use_lsgan: self.loss = nn.MSELoss()
    else: self.loss = nn.BCELoss()

  def get_target_tensor(self, input, target_is_real):
    target_tensor = None
    if target_is_real:
      create_label = ((self.real_label_var is None) or
                      (self.real_label_var.numel() != input.numel()))
      if create_label:
        real_tensor = self.Tensor(input.size()).fill_(self.real_label)
        self.real_label_var = Variable(real_tensor, requires_grad=False)
      target_tensor = self.real_label_var
    else:
      create_label = ((self.fake_label_var is None) or
                      (self.fake_label_var.numel() != input.numel()))
      if create_label:
        fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
        self.fake_label_var = Variable(fake_tensor, requires_grad=False)
      target_tensor = self.fake_label_var
    return target_tensor

  def __call__(self, input, target_is_real):
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    else:            
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      return self.loss(input[-1], target_tensor)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, use_lsgan=True, target_real_label=0.0, target_fake_label=1.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.use_lsgan = use_lsgan
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()


    def get_target_tensor(self,  label,prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        s = prediction.size()

        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1).cuda()
            target_tensor = torch.cat([label, real_label], dim=-1)

        else:
            fake_label = self.fake_label.expand(label.size(0), 1).cuda()
            target_tensor = torch.cat([label, fake_label], dim=-1)

        target_tensor = target_tensor.view(s[0], s[1], 1, 1)
        target_tensor = target_tensor.repeat(1, 1, s[2], s[3])
        return target_tensor

    def __call__(self, prediction, label, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.use_lsgan:
          if isinstance(prediction[0], list):
            loss = 0
            for input_i in prediction:
              pred = input_i[-1]
              target_tensor = self.get_target_tensor(label, prediction,target_is_real)
              loss += self.loss(pred, target_tensor)
            return loss
          else:
            target_tensor = self.get_target_tensor(label, prediction[-1], target_is_real)
            return self.loss(prediction[-1], target_tensor)     

        else:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
