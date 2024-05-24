import torch
import torch.nn as nn
import torch.nn.functional as F 

# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomLoss(nn.Module):
    def __init__(self, lambda_value,num_classes):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value
        self.num_classes=num_classes
        self.epsilon=torch.tensor(1e-8,device=device)

    def assgin_weights(self,probs,targets):
        
        position_difference = torch.arange(probs.size(1), device=device).unsqueeze(0) - targets.unsqueeze(1)
        weights=torch.ones_like(probs,device=device)
        weights = weights*position_difference
        
        return weights/self.num_classes
    
    def stable_softmax(self,x):
        max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        x_exp = torch.exp(x - max_vals)
        return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


    def forward(self, outputs, targets):
        
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Convert targets to one-hot encoding and get the probs for classess with a stable softmax function
        probs = self.stable_softmax(outputs)        
        
        # get higher log-probs than true prob
        true_probs=probs[torch.arange(probs.size(0)), targets]
        true_expanded = true_probs.unsqueeze(1).expand(-1, probs.size(1))
        mask = probs > true_expanded
        higher__log_probs=torch.log(self.epsilon+1-probs*mask)

        # make all values zero before the true class
        mask = torch.arange(higher__log_probs.size(1)).unsqueeze(0).to(device=device) > targets.unsqueeze(1)
        over_log_probs = higher__log_probs * mask.int()
        
        # assign weights to each overestiamted class
        weights=self.assgin_weights(over_log_probs,targets)
        
        # get weighted log-probs
        w_over_log_probs=weights*over_log_probs

        # compute total loss
        over_loss=torch.mean(torch.sum(w_over_log_probs,dim=1),dim=0)
        loss=ce_loss-self.lambda_value*over_loss
        
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
        self.convbn=nn.BatchNorm2d(self.output_channels)
        
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
        x = self.convbn(x)
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
