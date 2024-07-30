#ifndef LARGEBATCH_H
#define LARGEBATCH_H

#include <iostream>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;
const int lengthEpsilon=100;
const int channels=7;
const int width=4;
const int height=4;
const int classNum=8;


// gpu cores
std::vector<torch::Device> devices = {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)};

template <class action>
struct BatchworkUnit {
	action pre[workDepth];
	vector<action> solution;
	vector<int> gHistogram;
	vector<int> fHistogram;
	double nextBound;
	uint64_t expanded, touched;
	int unitNumber;
	int nodeCount;
	bool processing;
	int ID;
	int checked;
	int b;
};

template <class state,class action>
class LargeBatch {
public:
	LargeBatch(int size,int numworks);
	~LargeBatch();
	bool Add(vector<state>& cubestates, vector<int*>& indexes, BatchworkUnit<action>* work, int& numChildren,int& batch,
																int& sNum, int& wNum);
	bool GetStuck(int& batch,int& sNum, int& wNum);
	void Terminate(int& batch);
	void Switch(int& batch);
	vector<torch::TensorOptions> options;
	vector<torch::TensorOptions> options_long;
	vector<torch::Tensor> gpu_inputs,h_values,samples,outputs;
	vector<int*> units;
	vector<BatchworkUnit<action>*> worksInProcess;
	int mark,workMark;

private:
	int maxbatchsize,worksNum,whichBatch;
	vector<bool> receives,dones;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
};

template <class state, class action>
LargeBatch<state,action>::LargeBatch(int size,int nw)
:maxbatchsize(size),mark(0),workMark(0),worksNum(nw),receives{true,false},dones{false,true}
{	
	units.resize((size+lengthEpsilon)*2);
	worksInProcess.resize(nw*2);

	for(unsigned int x = 0; x < devices.size(); x++)
	{
		options.push_back(torch::TensorOptions().device(devices[x]).dtype(torch::kFloat32));
		options_long.push_back(torch::TensorOptions().device(devices[x]).dtype(torch::kInt64));
	}
	
	for(unsigned int x = 0; x < devices.size(); x++)
	{
		samples.push_back(torch::zeros({size+lengthEpsilon, channels, width, height}));

		gpu_inputs.push_back(torch::empty({size+lengthEpsilon, channels,width,height}, options[x]));
		outputs.push_back(torch::empty({size+lengthEpsilon, classNum}, options_long[x]));
		h_values.push_back(torch::empty({size+lengthEpsilon}, options_long[x]));
		
	}
	
}

template <class state, class action>
LargeBatch<state,action>::~LargeBatch()
{
}

template <class state, class action>
bool LargeBatch<state,action>::Add(vector<state>& cubestates, vector<int*>& indexes, 
									BatchworkUnit<action>* work, int& numChildren,int& batch,
									int& sNum, int& wNum)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (receives[0] || receives[1]) ;});
	
	if(receives[0])
		whichBatch=0;
	else
		whichBatch=1;

	assert(!(receives[0] && receives[1]));
	
	int unitsIndex=whichBatch*(maxbatchsize+lengthEpsilon);
	int worksIndex=whichBatch*worksNum;
	for(unsigned int i = 0; i < numChildren; i++)
	{
		// change batchunits in units
		units[unitsIndex+mark+i]=indexes[i];

		// edit samples with new states
	}
	worksInProcess[worksIndex+workMark]=work;
	
	
	if(mark==0)
	{
		work->checked=1;
		work->b=whichBatch;
	}	

	mark=mark+numChildren;
	workMark++;
	work->processing=true;

    if(mark>=maxbatchsize)
    {	
		batch=whichBatch;
		sNum=mark;
		wNum=workMark;
		
		receives[whichBatch]=false;
		dones[whichBatch]=false;
		mark=0;
		workMark=0;
	}

	return receives[whichBatch];

}

template <class state, class action>
void LargeBatch<state,action>::Switch(int& batch)
{
	std::unique_lock<std::mutex> l(lock);
	
	if((!receives[(batch+1)%2]) && dones[(batch+1)%2])
	{
		receives[(batch+1)%2]=true;
		Full.notify_all();
	}	
	
}

template <class state, class action>
bool LargeBatch<state,action>::GetStuck(int& batch,int& sNum, int& wNum)
{
	std::unique_lock<std::mutex> l(lock);
	
	if(dones[batch])
	{
		sNum=mark;
		wNum=workMark;
	
		receives[batch]=false;
		dones[batch]=false;
		mark=0;
		workMark=0;

		return true;
	}
	else
		return false;
	
}

template <class state, class action>
void LargeBatch<state,action>::Terminate(int& batch)
{
	std::unique_lock<std::mutex> l(lock);
	
	dones[batch]=true;
	if(!receives[(batch+1)%2])
	{
		receives[batch]=true;
		Full.notify_all();	
	}
}


#endif