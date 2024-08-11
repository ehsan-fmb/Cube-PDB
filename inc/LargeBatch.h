#ifndef LARGEBATCH_H
#define LARGEBATCH_H

#include <iostream>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include "MicroTimer.h"

using namespace std;

// gpu cores
std::vector<torch::Device> devices = {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)};

constexpr int lengthEpsilon=1000;
const int channels=7;
const int width=4;
const int height=4;
const int classNum=8;

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
};

template <class state,class action>
class LargeBatch {
public:
	LargeBatch(int size,int t,int nw,int gpu_core);
	~LargeBatch();
	void Add(vector<state>& cubestates, vector<int*>& indexes, BatchworkUnit<action>* fw);
	bool IsFull(int& wStart,int& uStart,int& wLength,int& uLength);
	torch::Tensor samples,h_values,gpu_input,narrow_cpu_tensor,gpu_slice,hcost_slice,tmp_slice;
	torch::TensorOptions options,options_long;
	at::cuda::CUDAStream stream1,stream2,stream3;
	vector<int*> units;
	vector<BatchworkUnit<action>*> worksInProcess;
	int mark,workMark,num;

private:
	int maxbatchsize,whichBatch,worksNum,timeout;
	vector<bool> receives;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
	torch::TensorAccessor<float, 4> samplesAccessor;
	MicroTimer timer;
};

template <class state, class action>
LargeBatch<state,action>::LargeBatch(int size,int t,int nw,int gpu_core)
:maxbatchsize(size),timeout(t),samples(torch::zeros({size+lengthEpsilon, channels, width, height})),
	samplesAccessor(samples.accessor<float,4>()),mark(0),workMark(0),worksNum(nw),receives{true,false},stream1(at::cuda::getStreamFromPool(false,gpu_core)), 
    stream2(at::cuda::getStreamFromPool(false,gpu_core)), stream3(at::cuda::getStreamFromPool(false,gpu_core)),num(gpu_core)
{	
	units.resize((size+lengthEpsilon)*2);
	worksInProcess.resize(nw*2);

	options = torch::TensorOptions().device(devices[gpu_core]).dtype(torch::kFloat32);
    options_long = torch::TensorOptions().device(devices[gpu_core]).dtype(torch::kInt64);
	
	gpu_input = torch::empty({size+lengthEpsilon, channels,width,height}, options);
	h_values = torch::empty({size+lengthEpsilon}, options_long);

	samples=samples.to(at::kHalf);
	gpu_input=gpu_input.to(at::kHalf);

}

template <class state, class action>
LargeBatch<state,action>::~LargeBatch()
{
}

template <class state, class action>
void LargeBatch<state,action>::Add(vector<state>& cubestates, vector<int*>& indexes, BatchworkUnit<action>* fw)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (receives[0] || receives[1]) ;});

	if(receives[0])
		whichBatch=0;
	else
		whichBatch=1;

	int unitsIndex=whichBatch*(maxbatchsize+lengthEpsilon);
	int worksIndex=whichBatch*worksNum;

	for(unsigned int i = 0; i < indexes.size(); i++)
	{
		// change batchunits in units
		units[unitsIndex+mark+i]=indexes[i];

		// edit samples with new states
	}

	worksInProcess[worksIndex+workMark]=fw;
	fw->processing=true;
	
	mark=mark+indexes.size();
	workMark=workMark+1;

    if(mark>=maxbatchsize)
    {	
		// timer.stopTimer(); // Stop the timer
		// cout<<"batch size: "<<mark<<'\n';
    	// std::cout << "Time taken for search: " << timer.getDuration() << " microseconds" << std::endl;
		// cout<<"*********************\n";

		receives[whichBatch]=false;
		Full.notify_all();
	}

}

template <class state, class action>
bool LargeBatch<state,action>::IsFull(int& wStart,int& uStart,int& wLength,int& uLength)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::microseconds(timeout), [this](){return (mark>=maxbatchsize);});
	
	if(mark>0)
	{
		receives[whichBatch]=false;
		timeout=5;

		wStart=whichBatch*worksNum;
		uStart=whichBatch*(maxbatchsize+lengthEpsilon);
		wLength=workMark;
		uLength=mark;
		
		// timer.startTimer();

		mark=0;
		workMark=0;
		whichBatch=(whichBatch+1)%2;
		receives[whichBatch]=true;

		Full.notify_all();
		return true;
	}	
	else
		return false;
}

#endif