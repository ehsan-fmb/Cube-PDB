from utils.dataloader import readPDB
from utils.dataloader import get_state
import argparse
from utils.train import run
from utils.model import ResnetModel
import torch.nn as nn
import torch
import numpy as np

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
    parser.add_argument('-train')
    parser.add_argument('-epochs')
    parser.add_argument('-lr')
    parser.add_argument('-batch_size')
    args = parser.parse_args()
    
    if args.train=="True":
        
        # load the dataset
        dataset=readPDB(args.pdb_name)
        print("dataset is loaded.")

        # initiate the models with xavier uniform weights
        model=ResnetModel((3,3),num_filters,filter_size,fc_dim,resnet_dim,resnet_blocks,out_dim,True)
        model.apply(initialize_weights)
        
        # start training
        run(model,dataset,float(args.lr),int(float(args.epochs)),int(float(args.batch_size))
            ,args.pdb_name,int(test_interval),accuracy_threshold,accuracy_decay)
    
    else:
        convert_model(args.pdb_name)
