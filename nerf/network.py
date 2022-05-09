import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers # number of layers in the sigma network
        self.hidden_dim = hidden_dim # size of hidden dimension
        self.geo_feat_dim = geo_feat_dim # size of output dimension (not including the 1 output that represents the spacial density)
        self.encoder, self.in_dim = get_encoder(encoding, N_max=2048 * bound, d=3, L=16, F=2, b=2, N_min=16, log2_hashmap_size=19, N_max=2048) # initialize the input encoder

        # initialize the sigma network
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim # the first layer gets in_dim size of inputs
            else:
                in_dim = hidden_dim # the hidden layers get hidden_dim size inputs
            
            if l == num_layers - 1: # if l is the last layer
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else: # if it is a hidden layer
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False)) # append linear layers

        self.sigma_net = nn.ModuleList(sigma_net) # create a module with the initialized layers

        # color network
        self.num_layers_color = num_layers_color # number of layers in the color network     
        self.hidden_dim_color = hidden_dim_color # size of hidden dimension
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir) # initialize the viewing direction input encoder
        self.in_dim_color += self.geo_feat_dim # add the shape of 15 SH features to the shape of the encoded viewing directions
        
        # initialize the color network
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim # the first layer gets in_dim size of inputs
            else:
                in_dim = hidden_dim # the hidden layers get hidden_dim size inputs
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False)) # append linear layers

        self.color_net = nn.ModuleList(color_net) # create a module with the initialized layers

    
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]; where N: batch size
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound) # encode the coordinates using multiresolution hashmap technique

        # The sigma network to predict the space density of a given voxel
        h = x
        for l in range(self.num_layers): # forward pass of the sigma network
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0]) # the first output is considered the space density
        geo_feat = h[..., 1:] # the rest of the output is the hidden representation

        # color
        
        d = self.encoder_dir(d) 
        # d: the view direction projected onto the 
            # first 16 coefficients of the spherical harmonics basis (frequency encoding over unit vectors)

        h = torch.cat([d, geo_feat], dim=-1) # concatenate the hidden representation obtained from the sigma network with the encoded viewing direction parameter
        for l in range(self.num_layers_color): # forward pass of the color network
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color # return the predicted spacial density and the color for the input voxels

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        