import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import pandas as pd
import partitioning
import sampling
import multi_hop_cuda
import neighbor_search
import shared_aggregation
import unique_aggregation



def uniform_sampling(xyz,npoint,step,block_size, xyz_offset):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    B, N, C = xyz.shape
    S = npoint
    start_event.record()
    u_order, u_len, u_offset, point2group= partitioning.partitioning(xyz.squeeze(0), step, block_size, xyz_offset)
    end_event.record()
    torch.cuda.synchronize()
    partitioning_time = start_event.elapsed_time(end_event)
    # print(f"partitioning time: {partitioning_time:.4f} ms")
    
    start_event.record()
    fps_idx, xyz_centers = sampling.parallel_strided_sampling(xyz, u_order, npoint) 
    end_event.record()
    torch.cuda.synchronize()
    sampling_time = start_event.elapsed_time(end_event)
    # print(f"sampling time: {sampling_time:.4f} ms")

    return fps_idx, xyz_centers, u_order, u_len, u_offset, point2group,partitioning_time,sampling_time


def neighbor_search_common (block_size, hop, nsample, radius,centers, point2group, u_len, u_offset, u_order, xyz):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    block_num = block_size**3
    search_size = 2*hop+1
    search_total = search_size**3
    npoint = centers.shape[0]
    centers = centers.to(torch.int32)

    B, N, C = xyz.shape

    searching_array, searching_length, searching_offset, len_per_group, valid_length = multi_hop_cuda.searching_array_kernel(
            block_size, block_num, hop, search_size, search_total, npoint, centers,
            point2group, u_len
        )
    
    start_event.record()
    have_center, _1center_in_group, shared_len, isn_shared, ns_index, ns_distance,neighbor_len = neighbor_search.neighbor_search(
            N, npoint, block_num, nsample,radius**2, xyz.squeeze(0), centers, point2group, u_len,
            u_offset, u_order, searching_array, searching_length, searching_offset, len_per_group, valid_length, search_total
        )
    end_event.record()
    torch.cuda.synchronize()
    neighbor_search_time = start_event.elapsed_time(end_event)
    # print(f"neighbor_search time: {neighbor_search_time:.4f} ms")
    
    return have_center, isn_shared, searching_length, searching_offset, _1center_in_group, ns_index, shared_len, neighbor_len,neighbor_search_time


class PointShufflerSetAbstraction(nn.Module):
    def __init__(self, npoint, block_size, hop,radius, nsample, in_channel, mlp, group_all):
        super(PointShufflerSetAbstraction, self).__init__()
        self.npoint = npoint
        self.block_size = block_size
        self.hop = hop
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        
        self.max_coordinate = 2.#点云坐标范围
        self.correction_factor = 0.0002
        if block_size != None:
            self.block_num = block_size ** 3
            self.step = self.max_coordinate / block_size +  self.correction_factor
        translation_value = 1.0011
        self.xyz_offset = partitioning.coord_offset()
        self.xyz_offset.x_offset = translation_value
        self.xyz_offset.y_offset = translation_value
        self.xyz_offset.z_offset = translation_value
        torch.set_printoptions(threshold=np.inf,precision=4,sci_mode=False)
        


    def forward(self, xyz, points,layer_id = None):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        _, _, N = xyz.shape

        xyz = xyz.permute(0, 2, 1)
        

        if points is not None:
            points = points.permute(0, 2, 1)


            new_points = torch.cat([xyz,points], dim=-1)  
        else:
            new_points = xyz
        
        new_points = new_points.permute(0, 2, 1).unsqueeze(-2) 
        
        start_event.record()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        end_event.record()
        torch.cuda.synchronize()
        feature_update_time = start_event.elapsed_time(end_event)
        # print(f"feature_update time: {feature_update_time:.4f} ms")
    

        if self.group_all:
            new_points = new_points.permute(0,1,3,2)     
            new_points = torch.max(new_points, 2)[0]          
            return xyz, new_points          

        else:

            fps_idx, xyz_centers, u_order, u_len, u_offset, point2group,partitioning_time,sampling_time = uniform_sampling(xyz, self.npoint, self.step, self.block_size, self.xyz_offset)

            have_center, isn_shared, searching_length, searching_offset,_1center_in_group, ns_index, shared_len, neighbor_len,neighbor_search_time = neighbor_search_common(self.block_size, self.hop, self.nsample, self.radius, fps_idx,point2group,u_len,u_offset,u_order,xyz)          

            new_points = new_points.squeeze(-2).squeeze(0).permute(1,0) 

            new_points = new_points.contiguous()
            
            start_event.record()
            gather_result = shared_aggregation.shared_aggregation(fps_idx,have_center, new_points, searching_offset, _1center_in_group, ns_index, shared_len, self.nsample, self.block_num)
            end_event.record()
            torch.cuda.synchronize()
            shared_aggregation_time = start_event.elapsed_time(end_event)
            # print(f"shared_aggregation time: {shared_aggregation_time:.4f} ms")

            start_event.record()
            sactter_result = unique_aggregation.unique_aggregation(fps_idx,point2group,new_points,ns_index,isn_shared,searching_length,searching_offset,gather_result,shared_len,self.nsample)
            end_event.record()
            torch.cuda.synchronize()
            unique_aggregation_time = start_event.elapsed_time(end_event)
            # print(f"unique_aggregation time: {unique_aggregation_time:.4f} ms")

            new_points = sactter_result.unsqueeze(0).permute(0, 2 ,1) 
            new_xyz = xyz_centers.unsqueeze(0).permute(0, 2, 1)

            if layer_id in [1, 2]:
                times = [partitioning_time, sampling_time, feature_update_time, neighbor_search_time, shared_aggregation_time, unique_aggregation_time]
                return new_xyz, new_points, times
            
        return new_xyz, new_points,times

 