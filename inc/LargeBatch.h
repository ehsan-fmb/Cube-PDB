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

constexpr int lengthEpsilon=500;
const int channels1=7;
const int channels2=8;
const int width=4;
const int height=4;
const int classNum=8;
constexpr int batchWorkDepth=14;

template <class action>
struct BatchworkUnit {
	action pre[batchWorkDepth];
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
	LargeBatch(int size,int t,int numworks);
	~LargeBatch();
	void Add(vector<state>& cubestates, vector<int*>& indexes, vector<BatchworkUnit<action>*>& works);
	bool IsFull(int& wStart,int& uStart,int& wLength,int& uLength);
	torch::Tensor samples_1,samples_2,samples_3,samples_4,h_values_1,h_values_2,h_values_3,h_values_4,gpu_input_1,gpu_input_2;
	torch::TensorOptions options,options_long;
	at::cuda::CUDAStream stream1,stream2,stream3;
	vector<int*> units;
	vector<BatchworkUnit<action>*> worksInProcess;
	int mark,workMark;

private:
	int maxbatchsize,whichBatch,worksNum,timeout;
	vector<bool> receives;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
	torch::TensorAccessor<at::Half, 4> samplesAccessor_1,samplesAccessor_2,samplesAccessor_3,samplesAccessor_4;
	MicroTimer timer;
	int state_new[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
};

template <class state, class action>
LargeBatch<state,action>::LargeBatch(int size,int t,int nw)
:maxbatchsize(size),timeout(t),samples_1(torch::zeros({size+lengthEpsilon, channels1, width, height},torch::dtype(at::kHalf))),
	samples_2(torch::zeros({size+lengthEpsilon, channels2, width, height},torch::dtype(at::kHalf))),
	samples_3(torch::zeros({size+lengthEpsilon, channels1, width, height},torch::dtype(at::kHalf))),
	samples_4(torch::zeros({size+lengthEpsilon, channels2, width, height},torch::dtype(at::kHalf))),
	samplesAccessor_1(samples_1.accessor<at::Half,4>()),samplesAccessor_2(samples_2.accessor<at::Half,4>()),
	samplesAccessor_3(samples_3.accessor<at::Half,4>()),samplesAccessor_4(samples_4.accessor<at::Half,4>()),
	mark(0),workMark(0),worksNum(nw),receives{true,false},stream1(at::cuda::getStreamFromPool(false,1)),
	stream2(at::cuda::getStreamFromPool(false,1)), stream3(at::cuda::getStreamFromPool(false,1))
{	
	units.resize((size+lengthEpsilon)*2);
	worksInProcess.resize(nw*2);

	options = torch::TensorOptions().device(devices[0]).dtype(torch::kFloat32);
    options_long = torch::TensorOptions().device(devices[0]).dtype(torch::kInt64);
	
	gpu_input_1 = torch::empty({size+lengthEpsilon, channels1,width,height}, options);
	gpu_input_2 = torch::empty({size+lengthEpsilon, channels2,width,height}, options);
	h_values_1 = torch::empty({size+lengthEpsilon}, options_long);
	h_values_2 = torch::empty({size+lengthEpsilon}, options_long);
	h_values_3 = torch::empty({size+lengthEpsilon}, options_long);
	h_values_4 = torch::empty({size+lengthEpsilon}, options_long);

	gpu_input_1=gpu_input_1.to(at::kHalf);
	gpu_input_2=gpu_input_2.to(at::kHalf);

	at::cuda::CUDAGuard device_guard(0);
}

template <class state, class action>
LargeBatch<state,action>::~LargeBatch()
{
}

template <class state, class action>
void LargeBatch<state,action>::Add(vector<state>& cubestates, vector<int*>& indexes, vector<BatchworkUnit<action>*>& works)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (receives[0] || receives[1]) ;});
	
	if(receives[0])
		whichBatch=0;
	else
		whichBatch=1;

	if(mark==0)
	{
		if(whichBatch==0)
		{
			at::fill_(samples_3, 0);
			at::fill_(samples_4, 0);
		}
		else
		{
			at::fill_(samples_1, 0);
			at::fill_(samples_2, 0);
		}
	}

	int unitsIndex=whichBatch*(maxbatchsize+lengthEpsilon);
	int worksIndex=whichBatch*worksNum;

	for(unsigned int i = 0; i < indexes.size(); i++)
	{
		// change batchunits in units
		units[unitsIndex+mark+i]=indexes[i];

		auto puzzleState=cubestates[i];

		for (int j=0; j<16; j++) 
		{
      		state_new[puzzleState.puzzle[j]] = j;
    	}

		// // edit samples with new states
		if(whichBatch==0)
		{
			for(int val=1; val<=7; val++) 
			{
				int idx = state_new[val];
				samplesAccessor_3[mark+i][val-1][idx/4][idx%4] = 1;
			}

			for(int val=8; val<=15; val++) 
			{
				int idx = state_new[val];
				samplesAccessor_4[mark+i][val-8][idx/4][idx%4] = 1;
			}
		}
		else
		{
			for(int val=1; val<=7; val++) 
			{
				int idx = state_new[val];
				samplesAccessor_1[mark+i][val-1][idx/4][idx%4] = 1;
			}

			for(int val=8; val<=15; val++) 
			{
				int idx = state_new[val];
				samplesAccessor_2[mark+i][val-8][idx/4][idx%4] = 1;
			}
		}		
	}

	for(unsigned int i = 0; i < works.size(); i++)
	{
		worksInProcess[worksIndex+workMark+i]=works[i];
		works[i]->processing=true;
	}
	
	mark=mark+indexes.size();
	workMark=workMark+works.size();

    if(mark>=maxbatchsize)
    {	
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