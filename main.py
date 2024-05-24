from utils.dataloader import readPDB
import argparse
from utils.train import run
from utils.model import ResnetModel
import sys
from utils.dataloader import get_state

# NN layers
fc_dim=5000
num_filters=64
filter_size=2
resnet_dim=1000
resnet_blocks=4
out_dim=12

# training hyperparamters
test_interval=1e3
accuracy_threshold=0.01
accuracy_decay=0.997



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

        # initiate the models
        model=ResnetModel((3,3),num_filters,filter_size,fc_dim,resnet_dim,resnet_blocks,out_dim,True)
        
        # start training
        run(model,dataset,float(args.lr),int(float(args.epochs)),int(float(args.batch_size))
            ,args.pdb_name,int(test_interval),accuracy_threshold,accuracy_decay)

    