from utils.dataloader import readPDB
from utils.dataloader import get_state
from utils.model import CustomLoss
from utils.train import test
import argparse
from utils.train import run
from utils.model import ResnetModel
import torch.nn as nn
import torch
import numpy as np
from utils.CornerPDB import State

# NN layers
fc_dim=5000
num_filters=64
filter_size=2
resnet_dim=1000
resnet_blocks=4
out_dim=12

# training hyperparamters
test_interval=1e3
accuracy_threshold=0.001
accuracy_decay=0.988

# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def evaluation(name):
    
    # load the model
    model=ResnetModel((3,3),num_filters,filter_size,fc_dim,resnet_dim,resnet_blocks,out_dim,True)
    model.load_state_dict(torch.load("models/"+name+"/"+'model.pth'))
    model.to(device)

    # load the dataset
    dataset=readPDB(name)
    print("dataset is loaded.")

    criterion =CustomLoss(1,model.out_dim)
    overestimation,sum=test(model,dataset,criterion,chunk_num=2000)
    
    with open("models/"+name+"/"+'info.txt', 'a') as file:
        file.write("*"*100+"\n")
        file.write("*"*100+"\n")
        file.write("Final info by using whole dataset as test set:"+"\n")
        file.write("average heuristic: "+str(sum)+"\n")
        file.write("overestimated states: "+str(overestimation)+"\n")
        file.write("overestimation rate: "+str(overestimation/len(dataset))+"\n")


def convert_model(name):
    
    # load the model
    model=ResnetModel((3,3),num_filters,filter_size,fc_dim,resnet_dim,resnet_blocks,out_dim,True)
    model.load_state_dict(torch.load("models/"+name+"/"+'model.pth'))
    model.eval()
    
    # Trace the model
    example_input=get_state(10)
    example_input=np.expand_dims(example_input, 0)
    example_input=torch.tensor(example_input,dtype=torch.float32)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("models/"+name+"/"+"model_traced.pt")
    


# Initialize the weights using Xavier uniform
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb_name')
    parser.add_argument('-task')
    parser.add_argument('-epochs')
    parser.add_argument('-lr')
    parser.add_argument('-batch_size')
    args = parser.parse_args()
    
    if args.task=="train":
        
        # load the dataset
        dataset=readPDB(args.pdb_name)
        print("dataset is loaded.")

        # initiate the models with xavier uniform weights
        model=ResnetModel((3,3),num_filters,filter_size,fc_dim,resnet_dim,resnet_blocks,out_dim,True)
        model.apply(initialize_weights)
        
        # start training
        run(model,dataset,float(args.lr),int(float(args.epochs)),int(float(args.batch_size))
            ,args.pdb_name,int(test_interval),accuracy_threshold,accuracy_decay)
    
    elif args.task=="convert":
        convert_model(args.pdb_name)
    elif args.task=="test":
        evaluation(args.pdb_name)
    else:
        raise ValueError("task is not defined.")




