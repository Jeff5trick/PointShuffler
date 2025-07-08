import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointShufflerSetAbstraction
import os
import csv

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointShufflerSetAbstraction(npoint=512, block_size = 10,hop=1,radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointShufflerSetAbstraction(npoint=128, block_size = 10,hop=1,radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointShufflerSetAbstraction(npoint=None, block_size = None,hop=None,radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)

        self.bn1 = nn.BatchNorm1d(512) 
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
        self.num = 0
        self.sa1_times = []
        self.sa2_times = []

    def forward(self, xyz):

        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points,times = self.sa1(xyz, norm, layer_id=1)
        if times is not None:
            self.sa1_times.append(times)

        l2_xyz, l2_points,times = self.sa2(l1_xyz, l1_points, layer_id=2)
        if times is not None:
            self.sa2_times.append(times)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points

    def save_times_to_csv(self):

        if self.sa1_times:
            with open('sa1_times.csv', 'w', newline='') as file:
                writer = csv.writer(file)

                writer.writerow(['partitioning time', 'sampling time', 'feature_update time', 'neighbor_search time', 'shared_aggregation time', 'unique_aggregation time'])

                for times in self.sa1_times:
                    writer.writerow(times)

                writer.writerow(['Average'])

                if len(self.sa1_times) > 2:
                    averages = [sum(col[i] for i in range(2, len(self.sa1_times))) / (len(self.sa1_times) - 2) for col in zip(*self.sa1_times)]
                    writer.writerow(averages)
                elif len(self.sa1_times) == 2:
                    writer.writerow(['N/A'] * 6)  
                else:
                    writer.writerow(['N/A'] * 6)  


        if self.sa2_times:
            with open('sa2_times.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['partitioning time', 'sampling time', 'feature_update time', 'neighbor_search time', 'shared_aggregation time', 'unique_aggregation time'])
                for times in self.sa2_times:
                    writer.writerow(times)
                writer.writerow(['Average'])
                
                if len(self.sa2_times) > 2:
                    averages = [sum(col[i] for i in range(2, len(self.sa2_times))) / (len(self.sa2_times) - 2) for col in zip(*self.sa2_times)]
                    writer.writerow(averages)
                elif len(self.sa2_times) == 2:
                    writer.writerow(['N/A'] * 6) 
                else:
                    writer.writerow(['N/A'] * 6)  

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
