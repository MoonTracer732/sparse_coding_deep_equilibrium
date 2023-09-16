#@title Dependencies and functions
# HQS return z image

from math import log10, sqrt
from ttictoc import tic,toc
from scipy.io import loadmat
import glob,os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
from pathlib import Path
from PIL import Image
import random
import numpy as np
import skimage
from functools import partial
import math
import cv2
import imagesc as imagesc
import torch
import torch.nn as nn
import scipy
from torch.autograd import Variable
from torch.nn import Linear, ReLU, LeakyReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import glob,os
import mat73
import random
from sklearn.linear_model import OrthogonalMatchingPursuit
from math import log10, sqrt
#from google.colab import drive
#from dask import dataframe as dd
#drive.mount("/content/drive/", force_remount=True)

# Set the GPu as default
device = torch.device("cuda")


#%% functions

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[2]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[:,:,ch]).cpu().numpy()
            y = torch.squeeze(Y[:,:,ch]).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwssim = Bandwise(partial(ssim))
cal_bwpsnr = Bandwise(partial(skimage.metrics.peak_signal_noise_ratio, data_range=255))


def cal_sam(X, Y, eps=1e-8):
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    tmp = (np.sum(X*Y, axis=2) + eps) / (np.sqrt(np.sum(X**2, axis=2)) + eps) / (np.sqrt(np.sum(Y**2, axis=2)) + eps)
    return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam

def read_test_hyperspectral_images(image_path, indx):
    os.chdir(image_path)
    mat_files = glob.glob('*.mat')
    fname = mat_files[indx]
    print(fname)
    matData = mat73.loadmat(fname)
    spectra = matData['rad']
    spectra = spectra[0:1100, 0:1100, :]
    #spectra = min_max_normalized(spectra)
    spectra = (spectra - np.amin(spectra))/(np.amax(spectra) - np.amin(spectra))
    return spectra

def myPSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = np.amax(original)
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def add_gaussian_noise_matrix_form(Hd, sigma):
    [N,M,B] = Hd.shape
    noise = sigma/255*np.random.randn(N, M, B);
    Hd_noisy = Hd + noise
    return Hd_noisy

def add_gaussian_noise_matrix_form_colored(Hd, sigma):
  [N,M,B] = Hd.shape
  Hd_noisy = np.zeros(Hd.shape)
  for i in range(Hd.shape[2]):
        sigma = [5, 10, 10, 30, 40, 60]
        noise = random.choice(sigma)/255*np.random.randn(N, M);
        Hd_noisy[:,:,i] = Hd[:,:,i] + noise

  return Hd_noisy


def add_gaussian_noise_matrix_form_colored_stripe(Hd, sigma):
  [N,M,B] = Hd.shape
  Hd_noisy = np.zeros(Hd.shape)
  for i in range(Hd.shape[2]):

        sigma = [5, 10, 10, 30, 40, 60]
        noise = random.choice(sigma)/255*np.random.randn(N, M);
        Hd_noisy[:,:,i] = Hd[:,:,i] + noise

  min_amount = 0.05;
  max_amount = 0.15;
  band = np.random.permutation(B);
  band = band[0:10];
  stripnum = np.random.randint(math.ceil(min_amount * N), math.ceil(max_amount * N), band.shape[0]);
  for i in range(band.shape[0]):
    loc = np.random.permutation(N);
    a = stripnum[i]
    loc = loc[1:a];
    stripe = np.random.rand(1, loc.shape[0])*0.5-0.25;
    Hd_noisy[:,loc,band[i]] = Hd_noisy[:,loc,band[i]] -stripe


  return Hd_noisy

def add_gaussian_noise_matrix_form_colored_deadline(Hd, sigma):
  [N,M,B] = Hd.shape
  Hd_noisy = np.zeros(Hd.shape)
  for i in range(Hd.shape[2]):

        sigma = [5, 10, 10, 30, 40, 60]
        noise = random.choice(sigma)/255*np.random.randn(N, M);
        Hd_noisy[:,:,i] = Hd[:,:,i] + noise

  min_amount = 0.05;
  max_amount = 0.15;
  band = np.random.permutation(B);
  band = band[0:10];
  stripnum = np.random.randint(math.ceil(min_amount * N), math.ceil(max_amount * N), band.shape[0]);
  for i in range(band.shape[0]):
    loc = np.random.permutation(N);
    a = stripnum[i]
    loc = loc[1:a];
    stripe = np.random.rand(1, loc.shape[0])*0.5-0.25;
    # Hd_noisy[:,loc,band[i]] = Hd_noisy[:,loc,band[i]] -stripe
    Hd_noisy[:,loc,band[i]] = 0

  return Hd_noisy

def construct_the_training_dataset(num_of_patches, patch_size, sparsity_level, D, image_path, noIm_start, noIm_end, sigma, gaus_sigma, rcond):
    # In this function random patches are extracted from a hyperspectral image
    # and its noisy version and the sparse coding matrices are estimated
    Ds_sup = []
    Noisy_data = []
    Clean_data = []
    cnt = 1
    for indx in range(noIm_start, noIm_end):
       print(indx)
       # Read a hyperspectral Image
       X = read_test_hyperspectral_images(image_path, indx)
      #  X_noisy = add_gaussian_noise_matrix_form(X, sigma)
      #  X_noisy = add_gaussian_noise_matrix_form_colored_stripe(X, sigma)
       X_noisy = add_gaussian_noise_matrix_form_colored_deadline(X, sigma)

       cnt, support_patch, data_patch, clean_patch = extract_from_random_patches_SCM(X, X_noisy,
                                                  num_of_patches, patch_size, sparsity_level, D, cnt, rcond)
       if indx == noIm_start:
         #G_data = g
         #G_noisy_data = g_noisy
         Ds_sup = support_patch
         Noisy_data = data_patch
         Clean_data = clean_patch
       else:
         #G_data = np.concatenate((G_data, g))
         #G_noisy_data = np.concatenate((G_noisy_data, g_noisy))
         Ds_sup = np.concatenate((Ds_sup, support_patch))
         Noisy_data = np.concatenate((Noisy_data, data_patch))
         Clean_data = np.concatenate((Clean_data, clean_patch))
       del X, X_noisy
    return  Ds_sup, Noisy_data, Clean_data

def extract_from_random_patches_SCM(X, X_noisy, num_of_patches, patch_size, sparsity_level, D, cnt, rcondition):
    # In this function random patches are extracted from a hyperspectral image
    # and its noisy version and the sparse coding matrices are estimate
    image_size = X.shape
    rI = random.sample(range(0, image_size[1]//patch_size-1), num_of_patches)
    rJ = random.sample(range(0, image_size[1]//patch_size-1), num_of_patches)

    #G2d_list = []
    #G2d_noisy_list = []
    dictionary_list = []
    data_list = []
    data_clean = []

    for xx in range(num_of_patches):
        for yy in range(num_of_patches):
            i = rI[xx]
            j = rJ[yy]
            x = X[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            x = (np.reshape(x, (patch_size*patch_size, image_size[2]))).T
            x_noisy = X_noisy[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            x_noisy = (np.reshape(x_noisy, (patch_size*patch_size, image_size[2]))).T
            #Normlaize the data
            #norm_x_noisy = (np.sum(x_noisy**2, axis=0))**0.5;
            #x_noisy = np.divide(x_noisy, norm_x_noisy)
            #x = np.divide(x, norm_x_noisy)
            #Find the support set based on OMP algorithm and the centroid signal
            x_noisy_mean = np.mean(x_noisy, axis=1)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity_level, fit_intercept=False, normalize=False)
            omp.fit(D, x_noisy_mean)
            coef = omp.coef_
            sup, = coef.nonzero()
            #g = np.dot(np.linalg.pinv(D[:,sup], rcond=rcondition, hermitian=False),x)
            #g_noisy = np.dot(np.linalg.pinv(D[:,sup], rcond=rcondition, hermitian=False),x_noisy)
            #g_noisy= np.ones((sparsity_level,patch_size*patch_size))
            #g = np.reshape(g, (sparsity_level, patch_size, patch_size))
            #g_noisy = np.reshape(g_noisy, (sparsity_level, patch_size, patch_size))
            #G2d_list.append(g)
            #G2d_noisy_list.append(g_noisy)
            dictionary_list.append(D[:,sup])
            data_list.append(x_noisy)
            data_clean.append(x)
            cnt = cnt + 1
    return   cnt, dictionary_list, data_list, data_clean

def reconstruct_hyperspectral_image(X, X_noisy, patch_size, sparsity_level, D, my_sc_model, rcond, mtv, device, blocks, batch_size):
    # In this function random patches are extracted from an hyperspectral image
    # and its noisy version and the sparse coding matrices are estimated
    image_size = X.shape
    num_of_patch_i = image_size[0]//patch_size
    num_of_patch_j = image_size[1]//patch_size
    hd=np.zeros((image_size[0], image_size[1], image_size[2]))
    Data = []
    Dictionary = []
    Norm = []
    for i in range(num_of_patch_i):
        for j in range(num_of_patch_j):
            x_noisy = X_noisy[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            x_noisy = (np.reshape(x_noisy, (patch_size*patch_size, image_size[2]))).T
            #Normlaize the data
            #norm_x_noisy = (np.sum(x_noisy**2, axis=0))**0.5;
            #Find the support set based on OMP algorithm and the centroid signal
            x_noisy_mean = np.mean(x_noisy, axis=1)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity_level, fit_intercept=False, normalize=False)
            omp.fit(D, x_noisy_mean)
            coef = omp.coef_
            sup, = coef.nonzero()
            # Forward model
            Ds = D[:,sup]
            data = x_noisy
            Dictionary.append(Ds)
            Data.append(data)
            #Norm.append(norm_x_noisy)
    del data
    Data = np.array(Data)
    Dictionary = np.array(Dictionary)
    #Norm = np.array(Norm)

    dataloader_Dictionary = DataLoader(torch.from_numpy(Dictionary), batch_size, shuffle=False)
    dataloader_Data = DataLoader(torch.from_numpy(Data), batch_size, shuffle=False)
    #dataloader_Norm = DataLoader(torch.tensor(Norm), batch_size, shuffle=False)
    del Data, Dictionary
    Rec_Im = []
    cc = 0
    for data, Ds in zip(dataloader_Data, dataloader_Dictionary):
        cc
        batch_size = len(data)
        #norm = norm.numpy()
        G = my_sc_model((data.float()).to(device), (Ds.float()).to(device), len(data), sparsity_level, patch_size, mtv, device, blocks)
        #norm = np.reshape(norm, (batch_size, 1, patch_size*patch_size))
        im = G.cpu().detach().numpy()
        del G
        torch.cuda.empty_cache()
        im = np.reshape(im, (batch_size, data.shape[1], patch_size*patch_size))
        if cc == 0:
           Rec_Im = im
        else:
           Rec_Im = np.append(Rec_Im, im, axis=0)
        cc = cc + 1
        del data, Ds
    cnt = 0
    for i in range(num_of_patch_i):
       for j in range(num_of_patch_j):
          hd[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,:]= np.reshape((Rec_Im[cnt,:,:]).T, (patch_size, patch_size, 31))
          cnt = cnt + 1
    del Rec_Im, dataloader_Dictionary, dataloader_Data
    return hd

#%% torch network

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d


class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )
        self.freeze_bn = True
        self.freeze_bn_affine = True

    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)

        out = out + residual

        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

#%% title Models and training functions
#@title Models and training functions
#-------------------------------------------------------------------------------#


class deep_unrolled_sc(Module):
    def __init__(self, DnCnn):
        super(deep_unrolled_sc, self).__init__()
        self.DnCnn = DnCnn
        self.alpha=nn.Parameter(torch.tensor(10.99), requires_grad=True)

    # Defining the forward pass
    def forward(self, data, dictionary, batch_size, sparsity_level, patch_size, mtv, device, blocks):

        DsTr = torch.transpose(dictionary,1,2)
        A = torch.matmul(DsTr, data)
        A = self.alpha*A
        G = torch.matmul(DsTr ,dictionary)
        DtD = torch.linalg.inv(G*self.alpha + G)
        #1
        #x=torch.reshape(x, (batch_size, sparsity_level, patch_size*patch_size))
        x=(torch.ones((batch_size, sparsity_level, patch_size*patch_size))).to(device)
        Z=(torch.ones((batch_size, dictionary.shape[1], patch_size*patch_size))).to(device)
        for kk in range(blocks):
            x=torch.matmul(DtD, (A+torch.matmul(DsTr, Z)))
            Z=torch.matmul(dictionary, x)
            Z=torch.reshape(Z,(batch_size, dictionary.shape[1], patch_size, patch_size))
            Z = self.DnCnn(Z)
            z1 = Z
            Z=torch.reshape(Z, (batch_size, dictionary.shape[1], patch_size*patch_size))
        #Output
        #x=torch.reshape(x, (batch_size, sparsity_level, patch_size, patch_size))
        del A, G, DtD, x, Z
        torch.cuda.empty_cache()
        return z1


#@title Pretraining DnCNN
class DnCnn_net(nn.Module):
       def __init__(self, num_of_layers, features, input_channels):
         super().__init__()
         kernel_size = 3
         padding = 1
         features = features
         channels = input_channels
         num_of_layers = num_of_layers
         layers = []
         layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
         layers.append(nn.ReLU(inplace=True))
         for _ in range(num_of_layers-2):
             layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
             #layers.append(nn.BatchNorm2d(features))
             layers.append(nn.ReLU())
         layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
         self.dncnn = (nn.Sequential(*layers))

       def forward(self, input):
           return self.dncnn(input)

def pretraining(DnCnn_model, num_epochs, learning_rate, dataloader_pretrain, D, device, l2_w,
                 sparsity_level, batch_size, patch_size):
     criterion = nn.MSELoss()
     #criterion = nn.L1Loss()
     optimizer = torch.optim.Adam(DnCnn_model.parameters(), lr=learning_rate, weight_decay=l2_w)

     for epoch in range(num_epochs):
        acc_loss = 0.
        for s, data_noisy, clean in dataloader_pretrain:
            # if batch_size>1:
            batch_size = len(data_noisy)
            data_noisy = (data_noisy.float()).to(device)
            data_noisy = torch.reshape(data_noisy, (batch_size, clean.shape[1], patch_size, patch_size))
            clean = (clean.float()).to(device)
            clean = torch.reshape(clean, (batch_size, clean.shape[1], patch_size, patch_size))
            # ===================forward=====================
            output = DnCnn_model(data_noisy)
            loss = torch.sqrt(criterion(output, clean))
            #loss = (criterion(output, clean))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
        print(acc_loss / len(dataloader_pretrain))


def train_my_sc_model(my_sc_model, num_epochs, learning_rate, dataloader_train, D, device, l2_w, sparsity_level, batch_size, patch_size, mtv, blocks):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(my_sc_model.parameters(),lr=learning_rate,weight_decay=l2_w)
    for epoch in range(num_epochs):
       acc_loss = 0.
       for Ds, data, clean in dataloader_train:
           if batch_size>1:
               batch_size=len(data)
           clean = (clean.float()).to(device)
           clean = torch.reshape(clean, (batch_size, Ds.shape[1], patch_size, patch_size))
           # ===================forward=====================
           output = my_sc_model((data.float()).to(device), (Ds.float()).to(device), batch_size, sparsity_level, patch_size, mtv, device, blocks)
           loss = torch.sqrt(criterion(output, clean))
           # ===================backward====================
           optimizer.zero_grad()
           loss.backward(retain_graph=True)
           optimizer.step()
           acc_loss += loss.item()
       print(acc_loss / len(dataloader_train))
    return my_sc_model

#%%  title Read Dicitonary and the training Data
#@title Read Dicitonary and the training Data
#MAIN
#Read the dictionary
matData=scipy.io.loadmat("./dictionary.mat")
D= matData['D']


#Parameters
image_path = "/content/drive/MyDrive/DATA_ICVL/Train"
num_of_patches = 10
patch_size = 70
sparsity_level = 12
noIm_start = 0
noIm_end = 50
sigma = 30
gaus_sigma = 0.0005
rcond = 1e-05

# create the training dataset
train_Support, train_Data, train_clean = construct_the_training_dataset(
    num_of_patches, patch_size, sparsity_level, D, image_path, noIm_start, noIm_end, sigma, gaus_sigma, rcond)

train_Support = np.float32(train_Support)
train_Datat = np.float32(train_Data)
train_clean = np.float32(train_clean)

# Create the dataloader
batch_size = 6
train_ds = TensorDataset(torch.from_numpy(train_Support), torch.from_numpy(train_Data), torch.from_numpy(train_clean))
dataloader_train = DataLoader(train_ds, batch_size, shuffle=True)
del train_ds
del train_Support, train_Data, train_clean

#%% 
#@title Train the model
l2_w = 1e-09
num_epochs = 400
learning_rate = 1e-3

# DnCnn_model = DnCnn_net(5, 256, 31).cuda()
DnCnn_model = MemNet(31, 256, 1, 1).to(device )

pretraining(DnCnn_model, num_epochs, learning_rate, dataloader_train, D, device, l2_w, sparsity_level,
                               batch_size, patch_size)

learning_rate = 1e-4

pretraining(DnCnn_model, num_epochs, learning_rate, dataloader_train, D, device, l2_w, sparsity_level,
                               batch_size, patch_size)


#%% Train the DU model
#@title Train the DU model
l2_w = 1e-10
num_epochs = 50
learning_rate = 1e-4
mtv = 0.01
blocks = 10
# patch_size = 50
my_sc_model = deep_unrolled_sc(DnCnn_model).cuda()
# my_sc_model=torch.load("/content/drive/MyDrive/Colab Notebooks/DEEP_UNROLLING_phd/models/my_du_fast_stripe.pt")

learning_rate = 1e-3
train_my_sc_model(my_sc_model, num_epochs, learning_rate, dataloader_train, D, device, l2_w, sparsity_level, batch_size, patch_size, mtv, blocks)

torch.save(my_sc_model,"/content/drive/MyDrive/Colab Notebooks/DEEP_UNROLLING_phd/models/my_du_fast_deadline_denet.pt")

learning_rate = 1e-4
train_my_sc_model(my_sc_model, num_epochs, learning_rate, dataloader_train, D, device, l2_w, sparsity_level, batch_size, patch_size, mtv, blocks)

torch.save(my_sc_model,"/content/drive/MyDrive/Colab Notebooks/DEEP_UNROLLING_phd/models/my_du_fast_deadline_denet.pt")

#my_sc_model=torch.load("/content/drive/MyDrive/Colab Notebooks/DEEP_UNROLLING_phd/models/my_du_fast_deadline2.pt")
#%% 
#@title Testing
sigma = 30
patch_size = 100
blocks = 10
#Testing
image_test_path = "/content/drive/MyDrive/DATA_ICVL/Test_colored"
psnr_n = []
sim_n = []
sam_n = []
time_rec = []
for kk in range (39):
    print(kk)
    #Test reconstruction
    X = read_test_hyperspectral_images(image_test_path, kk)
    X_noisy = add_gaussian_noise_matrix_form_colored(X, sigma)
    # X_noisy = add_gaussian_noise_matrix_form_colored_stripe(X, sigma)
    # X_noisy = add_gaussian_noise_matrix_form_colored_deadline(X, sigma)

    # X_noisy = scipy.ndimage.gaussian_filter(X_noisy, 0.005)
    tic()
    X_recon = reconstruct_hyperspectral_image(X, X_noisy, patch_size, sparsity_level, D, my_sc_model, rcond, 1, device, blocks, 10)
    time_rec.append((toc()))
    X_recon[X_recon>1]=1
    X_recon[X_recon<0]=0

    a, b, c = MSIQA(torch.from_numpy(X[20:1050-20, 20:1050-20, :]*255), torch.from_numpy(X_recon[20:1050-20, 20:1050-20, :]*255))
    psnr_n.append(a)
    sim_n.append(b)
    sam_n.append(c)

print(np.mean(psnr_n))
print(np.mean(sim_n))
print(np.mean(sam_n))
print(np.mean(time_rec))





# kk=10
# X_recon = np.rot90(X_recon, k=1, axes=(0, 1))
# fig = plt.figure(figsize=(15,10))
# plt.imshow(np.uint8(X_recon[:,:,23]*255), cmap='gray', aspect='auto')
#plt.imshow(np.uint8(X_noisy[:,:,23]*255), cmap='gray', aspect='auto')
#plt.imshow(np.uint8(X[:,:,23]*255), cmap='gray', aspect='auto')