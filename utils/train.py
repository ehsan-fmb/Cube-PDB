import torch
import numpy as np
import torch.amp
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from utils.dataloader import get_state
from utils.model import CustomLoss
from multiprocessing import Pool
from torch.optim.lr_scheduler import StepLR
import gc

# cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define these terms
# lambda for second term of the loss
# pool for multiprocessing
# number samples for test
chunk_num=100
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

    return miss,torch.sum(classes)

def create_validation_set(dataset):
    lower_than_7=dataset[dataset[:,0]<7]
    rank_7=dataset[dataset[:,0]==7]
    rank_8=dataset[dataset[:,0]==8]
    rank_9=dataset[dataset[:,0]==9]
    rank_10=dataset[dataset[:,0]==10]
    rank_11=dataset[dataset[:,0]==11]

    rank_7=rank_7[np.random.choice(rank_7.shape[0], size=int(25e4), replace=False)]
    rank_8=rank_8[np.random.choice(rank_8.shape[0], size=int(5e5), replace=False)]
    rank_9=rank_9[np.random.choice(rank_9.shape[0], size=int(5e5), replace=False)]
    rank_10=rank_10[np.random.choice(rank_10.shape[0], size=int(5e5), replace=False)]
    
    validation_set=np.vstack((lower_than_7,rank_7,rank_8,rank_9,rank_10,rank_11))

    return validation_set,np.sum(validation_set[:,0])/validation_set.shape[0]

def test(model,dataset):
    
    # turn on testing mode
    model.eval()

    # slice the dataset to equal chunks
    sublist_length = len(dataset) // chunk_num
    remainder = len(dataset) % chunk_num
    total_miss=0
    total_sum=0

    start_index = 0
    for i in range(chunk_num):
        end_index = start_index + sublist_length - 1
        if i < remainder:
            end_index += 1
        samples=dataset[start_index:end_index+1,:]

        # get the overestimated samples for this chunk
        miss,heuristic_sum=calculate_overestimated_samples(model,samples)
        total_miss+=miss
        total_sum+=heuristic_sum

        start_index = end_index + 1

    return total_miss/len(dataset),total_sum/len(dataset)


def update_target(model,pdb_name):

    # save both original model
    torch.save(model.state_dict(),"models/"+pdb_name+"/"+'model.pth')



def make_batch(dataset=None,batch_size=None,test_samples=None):
    
    if test_samples is not None:
        samples=test_samples
    else:
        batch_size_0=batch_size//10
        batch_size_1=batch_size-batch_size_0
        start_index_0 = np.random.randint(0, len(dataset[0]) - batch_size_0 + 1)
        start_index_1=  np.random.randint(0, len(dataset[1]) - batch_size_1 + 1)
        samples_0 = dataset[0][start_index_0:start_index_0 + batch_size_0,:]
        samples_1 = dataset[1][start_index_1:start_index_1 + batch_size_1,:]
        samples=np.vstack((samples_0,samples_1))
    
    # use multiprocessing for speed
    inputs = pool.map(get_state,(index[1] for index in samples))

    # convert the results to torch tensors
    outputs=samples[:,0]
    inputs=np.stack(inputs)
    outputs=torch.tensor(outputs,device=device,dtype=torch.long)
    inputs=torch.tensor(inputs,device=device,dtype=torch.float32)
    
    
    return inputs,outputs

def display_progress(miss,losses,pdb_name,last_epoch,interval,average_heuristic,new_inaccuracy,new_average_heuristic,true_average=None):
    
    if losses:
        min_loss=min(losses)
        max_loss=max(losses)
        avg_loss=sum(losses)/len(losses)
    
        with open("models/"+pdb_name+"/"+'info.txt', 'a') as file:
            file.write("*"*50+"\n")
            file.write("Next interval: from "+str(last_epoch-interval)+" to "+str(last_epoch)+"\n")
            file.write("min loss: "+str(min_loss.item())+"\n")
            file.write("max loss: "+str(max_loss.item())+"\n")
            file.write("average loss: "+str(avg_loss.item())+"\n")
            file.write("new inaccuracy: "+str(new_inaccuracy.item())+"\n")
            file.write("new average heuristic: "+str(new_average_heuristic.item())+"\n")
            file.write("final inaccuracy: "+str(miss.item())+"\n")
            file.write("final average heuristic: "+str(average_heuristic.item())+"\n")

    else:
        with open("models/"+pdb_name+"/"+'info.txt', 'a') as file:
            file.write("Before training: "+"\n")
            file.write("inaccuracy: "+str(miss.item())+"\n")
            file.write("average heuristic :"+str(average_heuristic.item())+"\n")
            file.write("true average heuristic :"+str(true_average.item())+"\n")



def update(dataset_1,dataset_2,model,batch_size,optimizer,criterion):
        
    # get a uniformly random batch of data
    inputs,cost_to_go=make_batch(dataset=[dataset_1,dataset_2],batch_size=batch_size)
    
    # forward to get nn outputs
    nn_probs=model(inputs)
    
    #loss
    loss = criterion(nn_probs, cost_to_go)

    # backwards
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    
    # step
    optimizer.step()

    # refresh cuda memory
    del inputs
    del nn_probs
    del cost_to_go
    torch.cuda.empty_cache()
    gc.collect()
    
    return loss


def compare_models(inaccuracy,new_inaccuracy,avg_heuristic,new_avg_heuristic,accuracy_threshold):
    
    if new_inaccuracy+accuracy_threshold<inaccuracy:
        return True
    if abs(new_inaccuracy-inaccuracy)<accuracy_threshold and new_avg_heuristic>avg_heuristic:
        return True
    return False


def split_dataset(dataset):

    lower_than8=dataset[dataset[:,0]<8]
    dataset_rest=dataset[dataset[:,0]>=8]

    return lower_than8,dataset_rest

def run(model,dataset,learning_rate,epochs,batch_size,pdb_name,test_interval,accuracy_threshold,accuracy_decay):
    
    # define loss, optimizer, and send model paramters to the device
    model.to(device)
    criterion =CustomLoss(loss_lambda,model.out_dim)
    optimizer: Optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9998)
    
    # define the validation set and write its information
    val_dataset,true_average=create_validation_set(dataset)
    print("validation set is defined")

    # split the dataset to two sections: lower than 8 and thre rest
    dataset_1,dataset_2=split_dataset(dataset)
    
    # get the first inaccuracy before training has started
    inaccuracy=torch.tensor(1,device=device)
    avg_heuristic=torch.tensor(0,device=device)
    display_progress(inaccuracy,None,pdb_name,0,test_interval,avg_heuristic,None,None,true_average=true_average)

    # copy model paramters to the target model 
    update_target(model,pdb_name)
    
    losses=[]
    for i in range(epochs):
        
        # turn on training mode
        model.train()

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # do update for a specific number of batches
        loss=update(dataset_1,dataset_2,model,batch_size,optimizer,criterion)
        losses.append(loss)

        # update the learning rate scheduler
        scheduler.step()

        # update target model if model is improved
        if (i+1)%test_interval==0:    
            
            print("epoch: "+str(i+1))

            # get the new accuracy
            new_inaccuracy,new_average_heuristic=test(model,val_dataset)
            
            # update the model if a better accuracy is found
            if compare_models(inaccuracy,new_inaccuracy,avg_heuristic,new_average_heuristic,accuracy_threshold):
                inaccuracy=new_inaccuracy
                avg_heuristic=new_average_heuristic
                update_target(model,pdb_name)

            # decay the accuracy threshold
            accuracy_threshold=accuracy_threshold*accuracy_decay
            
            # print information in the file
            display_progress(inaccuracy,losses,pdb_name,(i+1),test_interval,avg_heuristic,new_inaccuracy,new_average_heuristic)
            losses=[]



     
     
          