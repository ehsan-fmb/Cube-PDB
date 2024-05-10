import torch
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from utils.dataloader import get_state
from utils.model import CustomLoss
from multiprocessing import Pool
import time
import gc


# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define these terms
# lambda for second term of the loss
# pool for multiprocessing
# number of chunks for test
chunk_num=1200
loss_lambda=1
num_process=None
pool = Pool(processes=num_process)


def calculate_overestimated_samples(model,samples):
    
    # get one-hot encoded inputs in parallel
    inputs,cost_to_go=make_batch(test_samples=samples)
    
    # get the probs from model
    nn_output=model(inputs)
    probs = torch.softmax(nn_output, dim=1)

    # pick the class from probs and calculte the overestimations
    classes=torch.argmax(probs,dim=1)    
    miss=torch.sum(cost_to_go<classes)

    # refresh cuda memory
    del inputs
    del nn_output
    del cost_to_go
    torch.cuda.empty_cache()
    gc.collect()

    return miss


def test(model,dataset):
    
    # slice the dataset to equal chunks
    sublist_length = len(dataset) // chunk_num
    remainder = len(dataset) % chunk_num
    miss=0

    start_index = 0
    for i in range(chunk_num):
        end_index = start_index + sublist_length - 1
        if i < remainder:
            end_index += 1
        samples=dataset[start_index:end_index+1,:]

        # get the overestimated samples for this chunk
        miss+=calculate_overestimated_samples(model,samples)

        start_index = end_index + 1

    return miss/len(dataset)


def update_target(model,target,pdb_name):
    for target_param, param in zip(target.parameters(), model.parameters()):
        target_param.data.copy_(param.data)
    
    # save the target model
    torch.save(target.state_dict(),"models/"+pdb_name+"/"+'model.pth')


def make_batch(dataset=None,batch_size=None,test=False,test_samples=None):
    
    if test_samples is not None:
        samples=test_samples
    else:
        start_index = np.random.randint(0, len(dataset) - batch_size + 1)
        samples = dataset[start_index:start_index + batch_size,:]
    
    # use multiprocessing for speed
    inputs = pool.map(get_state,(index[1] for index in samples))

    # convert the results to torch tensors
    outputs=samples[:,0]
    inputs=np.stack(inputs)
    outputs=torch.tensor(outputs,device=device,dtype=torch.long)
    inputs=torch.tensor(inputs,device=device,dtype=torch.float32)
    
    
    return inputs,outputs

def display_progress(miss,losses,pdb_name,last_epoch,interval):
    
    min_loss=min(losses)
    max_loss=max(losses)
    avg_loss=sum(losses)/len(losses)
    
    with open("models/"+pdb_name+"/"+'info.txt', 'a') as file:
        file.write("*"*50+"\n")
        file.write("Next interval: from "+str(last_epoch-interval)+" to "+str(last_epoch)+"\n")
        file.write("min loss: "+str(min_loss)+"\n")
        file.write("max loss: "+str(max_loss)+"\n")
        file.write("average loss: "+str(avg_loss)+"\n")
        file.write("inaccuracy:"+str(miss)+"\n")


def update(dataset,model,batch_size,optimizer,criterion):
        
    # get a uniformly random batch of data
    inputs,cost_to_go=make_batch(dataset=dataset,batch_size=batch_size)

    # forward to get nn outputs
    nn_probs=model(inputs)

    #loss
    loss = criterion(nn_probs, cost_to_go)

    # backwards
    loss.backward()

    # step
    optimizer.step()

    # refresh cuda memory
    del inputs
    del nn_probs
    del cost_to_go
    torch.cuda.empty_cache()
    gc.collect()
    
    return loss


def run(model,target_model,dataset,learning_rate,epochs,batch_size,pdb_name,test_interval):
    
    # define loss, optimizer, and send model paramters to the device
    model.to(device)
    target_model.to(device)
    criterion =CustomLoss(loss_lambda,model.out_dim)
    optimizer: Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Make the test batch and get the first inaccuracy
    inaccuracy=1

    # copy model paramters to the target model 
    update_target(model,target_model,pdb_name)

    model.train()
    losses=[]
    for i in range(epochs):
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # do update for a specific number of batches
        loss=update(dataset,model,batch_size,optimizer,criterion)
        losses.append(loss)

        # update target model if model is improved
        if (i+1)%test_interval==0:    
            
            print("epoch: "+str(i+1))

            # get the new accuracy
            new_inaccuracy=test(model,dataset)
            
            # update the model if a better accuracy is found
            if new_inaccuracy<inaccuracy:
                inaccuracy=new_inaccuracy
                update_target(model,target_model,pdb_name)
            
            # print information in the file
            display_progress(inaccuracy,losses,pdb_name,(i+1),test_interval)
            losses=[]



     
     
          