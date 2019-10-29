import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.nn import init

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width, depth):
        x_t = torch.ones([height, width, depth])
        x_t = torch.mul(x_t, torch.reshape(torch.range(0, height - 1),[height, 1, 1]))
        y_t = torch.ones([height, width, depth])
        y_t = torch.mul(y_t, torch.reshape(torch.range(0, width - 1), [1, width, 1]))
        z_t = torch.ones([height, width, depth])
        z_t = torch.mul(z_t, torch.reshape(torch.range(0, depth - 1), [1, 1, depth]))


        x_t = x_t.expand([height, width, depth])
        y_t = y_t.expand([height, width, depth])
        z_t = z_t.expand([height, width, depth])

        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()
            z_t = z_t.cuda()

        return x_t, y_t, z_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats),1),1,0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1,1)),rep)
        if self.use_gpu:
            x = x.cuda()

        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y, z):

        im = F.pad(im, (0,0,1,1,1,1,1,1,0,0))

        n_batch, height, width, depth, n_channel = im.shape

        n_batch, out_height, out_width, out_depth = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)

        x = x + 1
        y = y + 1
        z = z + 1

        max_x = height - 1
        max_y = width - 1
        max_z = depth - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        base = self.repeat(torch.arange(0, n_batch) * height * width * depth, out_height * out_width * out_depth)

        base_x0 = base + x0 * width * depth
        base_x1 = base + x1 * width * depth
        base00 = base_x0 + y0 * depth
        base01 = base_x0 + y1 * depth
        base10 = base_x1 + y0 * depth
        base11 = base_x1 + y1 * depth

        index000 = base00 + z0
        index001 = base00 + z1
        index010 = base01 + z0
        index011 = base01 + z1
        index100 = base10 + z0
        index101 = base10 + z1
        index110 = base11 + z0
        index111 = base11 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, n_channel])
        im_flat = im_flat.float()
        dim, _ = index000.transpose(1, 0).shape
        I000 = torch.gather(im_flat, 0, index000.transpose(1, 0).expand(dim, n_channel))
        I001 = torch.gather(im_flat, 0, index001.transpose(1, 0).expand(dim, n_channel))
        I010 = torch.gather(im_flat, 0, index010.transpose(1, 0).expand(dim, n_channel))
        I011 = torch.gather(im_flat, 0, index011.transpose(1, 0).expand(dim, n_channel))
        # print(torch.max(index100))
        # print(im_flat.shape)
        I100 = torch.gather(im_flat, 0, index100.transpose(1, 0).expand(dim, n_channel))
        I101 = torch.gather(im_flat, 0, index101.transpose(1, 0).expand(dim, n_channel))
        I110 = torch.gather(im_flat, 0, index110.transpose(1, 0).expand(dim, n_channel))
        I111 = torch.gather(im_flat, 0, index111.transpose(1, 0).expand(dim, n_channel))

        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        w111 = ((1.0-dx)*(1.0-dy)*(1.0-dz))
        w110 = ((1.0-dx)*(1.0-dy)*dz)
        w101 = ((1.0 - dx) * dy * (1.0 - dz))
        w100 = ((1.0 - dx) * dy * dz)
        w011 = (dx * (1.0 - dy) * (1.0 - dz))
        w010 = (dx * (1.0 - dy) * dz)
        w001 = (dx * dy * (1.0 - dz))
        w000 = (dx * dy * dz)

        w111 = w111.permute(1, 0)
        w110 = w110.permute(1, 0)
        w101 = w101.permute(1, 0)
        w100 = w100.permute(1, 0)
        w011 = w011.permute(1, 0)
        w010 = w010.permute(1, 0)
        w001 = w001.permute(1, 0)
        w000 = w000.permute(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                                                      w100 * I100, w101 * I101, w110 * I110, w111 * I111], dim = 1)), 1)
        output = torch.reshape(output,[n_batch, out_height, out_width, out_depth, n_channel])

        output = output.permute(0,4,1,2,3)

        return output

    def forward(self, moving_image, deformation_matrix):
        ## moving_image: size [batchsize,channels,x,y,z]
        ## deformation_matrix: size [batchsize,3,x,y,z]
        moving_image = moving_image.permute(0,2,3,4,1)
        deformation_matrix = deformation_matrix.permute(0, 2, 3, 4, 1)

        dx = deformation_matrix[:, :, :, :, 0]
        dy = deformation_matrix[:, :, :, :, 1]
        dz = deformation_matrix[:, :, :, :, 2]

        n_batch, height, width, depth = dx.shape

        x_mesh, y_mesh, z_mesh = self.meshgrid(height, width, depth)

        x_mesh = x_mesh.expand([n_batch, height, width, depth])
        y_mesh = y_mesh.expand([n_batch, height, width, depth])
        z_mesh = z_mesh.expand([n_batch, height, width, depth])

        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self.interpolate(moving_image, x_new, y_new, z_new)
