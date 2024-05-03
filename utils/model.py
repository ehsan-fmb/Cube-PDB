import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys

class CustomLoss(nn.Module):
    def __init__(self, lambda_value,num_classes):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value
        self.num_classes=num_classes

    def assgin_weights(self,probs,split):
        
        if split=="lower":
            weights = [(torch.tensor(len(prob))-torch.arange(len(prob))) for prob in probs]
            weights = [(weight/self.num_classes) for weight in weights]
        elif split=="higher":
            weights=[torch.ones_like(prob) for prob in probs]
        
        return weights


    def forward(self, outputs, targets):
        
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Convert targets to one-hot encoding and get the probs for classess
        probs = torch.softmax(outputs, dim=1)

        # split the probs in two parts: Underestimation and Overestimation (1-log-probs)
        lower_log_probs = [torch.log(1-probs[i, :index.item()]) for i, index in enumerate(targets)]
        higher_log_probs = [torch.log(1-probs[i, index.item() + 1:]) for i, index in enumerate(targets)]

        # assign the weights
        lower_weights=self.assgin_weights(lower_log_probs,split="lower")
        higher_weights=self.assgin_weights(higher_log_probs,split="higher") 

        # Calculate penalty term
        under_loss=torch.mean(torch.stack([torch.sum(lower_weight*lower_log_prob,dim=0) for lower_weight,lower_log_prob in 
                               zip(lower_weights,lower_log_probs)]))
        over_loss=torch.mean(torch.stack([torch.sum(higher_weight*higher_log_prob,dim=0) for higher_weight,higher_log_prob in 
                              zip(higher_weights,higher_log_probs)]))
        
        loss = ce_loss - self.lambda_value * (over_loss+under_loss)
        
        return loss


class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, output_channels: int, kernel_size:int  , h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.output_channels: int = output_channels
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm
        self.out_dim=out_dim

        
        # one convolutional layer
        self.conv1=nn.Conv2d(in_channels=36,out_channels=self.output_channels,kernel_size=kernel_size,stride=1)
        
        
        height=(state_dim[0]-kernel_size)+1
        width=(state_dim[1]-kernel_size)+1
        num_neurons=output_channels*height*width

        # first two hidden layers
        self.fc1 = nn.Linear(num_neurons, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for _ in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, self.out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # convolutional layer
        x = self.conv1(x)
        x = F.relu(x)
        
        # flatten the output
        x = x.view(x.size(0), -1)

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x
