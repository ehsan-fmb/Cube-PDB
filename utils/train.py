import torch
from torch import nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import random
from utils.dataloader import get_state
from utils.model import CustomLoss
import sys

# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lambda=1


def test(dataset,model,test_size):
    
    # get the test batch
    inputs,cost_to_go=make_batch(dataset,test_size)
    inputs= torch.stack(inputs)
    cost_to_go=torch.stack(cost_to_go,dim=0)
    
    # get the probs from model
    nn_output=model(inputs)
    probs = torch.softmax(nn_output, dim=1)

    # picl the class from probs and calculte the overestimations
    classes=torch.argmax(probs,dim=1)    
    miss=torch.sum(cost_to_go<classes)

    return miss/test_size


def update_target(model,target,pdb_name):
    for target_param, param in zip(target.parameters(), model.parameters()):
        target_param.data.copy_(param.data)
    
    # save the target model
    torch.save(target.state_dict(),"models/"+pdb_name+"/"+'model.pth')


def make_batch(dataset,batch_size):
    inputs=[]
    outputs=[]
    samples=random.sample(dataset, batch_size)
    
    for sample in samples:
        outputs.append(torch.tensor(sample[0], device=device,dtype=torch.long))
        state=get_state(sample[1])
        nn_input=state.get_nn_input()
        inputs.append(torch.tensor(nn_input,device=device,dtype=torch.float32))
    
    return inputs,outputs

def update(dataset,model,batch_size,optimizer,criterion):
        
    # get a uniformly random batch of data
    inputs,cost_to_go=make_batch(dataset,batch_size)
    
    # Add a batch dimension to the tensor
    inputs= torch.stack(inputs)
    cost_to_go=torch.stack(cost_to_go,dim=0)

    # forward to get nn outputs
    nn_probs=model(inputs)

    #loss
    loss = criterion(nn_probs, cost_to_go)

    # backwards
    loss.backward()

    # step
    optimizer.step()
    
    return loss


def run(model,target_model,dataset,learning_rate,epochs,batch_size,pdb_name,test_size,test_interval):
    
    # define loss, optimizer, and send model paramters to the device
    model.to(device)
    target_model.to(device)
    criterion =CustomLoss(loss_lambda,model.out_dim)
    optimizer: Optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #first test
    inaccuracy=test(dataset,model,test_size)
    print("First test is done.")

    # copy model paramters to the target model 
    update_target(model,target_model,pdb_name)

    model.train()
    for i in range(epochs):
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # do update for a specific number of batches
        loss=update(dataset,model,batch_size,optimizer,criterion)
        
        # display progress
        print("*"*40)
        print("epoch: "+str(i+1))
        print("average loss: "+ str(loss))

        # update target model if model is improved
        if (i+1)%test_interval==0:
            new_inaccuracy=test(dataset,model,test_size)
            if new_inaccuracy<inaccuracy:
                print("A new best model->")
                inaccuracy=new_inaccuracy
                update_target(model,target_model,pdb_name)


     
     
          