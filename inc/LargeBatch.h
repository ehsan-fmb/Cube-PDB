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
const int channels=36;
const int width=3;
const int height=3;
const int classNum=12;
constexpr int batchWorkDepth=5;

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
	LargeBatch(int size,int t,int nw,int gpu_core);
	~LargeBatch();
	void Add(vector<state>& cubestates, vector<int*>& indexes, vector<BatchworkUnit<action>*>& works);
	bool IsFull(int& wStart,int& uStart,int& wLength,int& uLength);
	int GetFaceColor(int face,state& s);
	torch::Tensor samples,h_values,gpu_input,narrow_cpu_tensor,gpu_slice,hcost_slice,tmp_slice,samples_test;
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
	torch::TensorAccessor<float, 4> samplesAccessor,samplesTestAccessor;
	MicroTimer timer;
};

template <class state, class action>
LargeBatch<state,action>::LargeBatch(int size,int t,int nw,int gpu_core)
:maxbatchsize(size),timeout(t),samples(torch::zeros({size+lengthEpsilon, channels, width, height})),
	samplesAccessor(samples.accessor<float,4>()),mark(0),workMark(0),worksNum(nw),receives{true,false},stream1(at::cuda::getStreamFromPool(false,gpu_core)), 
    stream2(at::cuda::getStreamFromPool(false,gpu_core)), stream3(at::cuda::getStreamFromPool(false,gpu_core)),num(gpu_core),
	samples_test(torch::zeros({size+lengthEpsilon, channels, width, height})),samplesTestAccessor(samples_test.accessor<float,4>())
{	
	units.resize((size+lengthEpsilon)*2);
	worksInProcess.resize(nw*2);

	options = torch::TensorOptions().device(devices[gpu_core]).dtype(torch::kFloat32);
    options_long = torch::TensorOptions().device(devices[gpu_core]).dtype(torch::kInt64);
	
	gpu_input = torch::empty({size+lengthEpsilon, channels,width,height}, options);
	h_values = torch::empty({size+lengthEpsilon}, options_long);

	samples=samples.to(at::kHalf);
	gpu_input=gpu_input.to(at::kHalf);

	at::cuda::CUDAGuard device_guard(1);
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

	int unitsIndex=whichBatch*(maxbatchsize+lengthEpsilon);
	int worksIndex=whichBatch*worksNum;

	for(unsigned int k = 0; k < indexes.size(); k++)
	{
		// change batchunits in units
		units[unitsIndex+mark+k]=indexes[k];

		// edit samples with new states
		for(int i = 0; i < 6; i++)
		{
			samplesTestAccessor[mark+k][7*i][1][1]=1;
			samplesTestAccessor[mark+k][7*i][0][1]=1;
			samplesTestAccessor[mark+k][7*i][1][0]=1;
			samplesTestAccessor[mark+k][7*i][1][2]=1;
			samplesTestAccessor[mark+k][7*i][2][1]=1;
		}

		// color corner cubies
		for(int i = 0; i < 8; i++)
		{
			if(i==0)
			{
				samplesTestAccessor[mark+k][GetFaceColor(0,cubestates[k])][2][0]=1;
				samplesTestAccessor[mark+k][12+GetFaceColor(2,cubestates[k])][0][0]=1;
				samplesTestAccessor[mark+k][24+GetFaceColor(1,cubestates[k])][0][2]=1;
			}
			else if(i==1)
			{
				samplesTestAccessor[mark+k][GetFaceColor(3,cubestates[k])][2][2]=1;
				samplesTestAccessor[mark+k][12+GetFaceColor(4,cubestates[k])][0][2]=1;
				samplesTestAccessor[mark+k][30+GetFaceColor(5,cubestates[k])][0][0]=1;
			}
			else if(i==2)
			{
				samplesTestAccessor[mark+k][GetFaceColor(6,cubestates[k])][0][2]=1;
				samplesTestAccessor[mark+k][30+GetFaceColor(7,cubestates[k])][0][2]=1;
				samplesTestAccessor[mark+k][18+GetFaceColor(8,cubestates[k])][0][0]=1;

			}
			else if(i==3)
			{
				samplesTestAccessor[mark+k][GetFaceColor(9,cubestates[k])][0][0]=1;
				samplesTestAccessor[mark+k][24+GetFaceColor(11,cubestates[k])][0][0]=1;
				samplesTestAccessor[mark+k][18+GetFaceColor(10,cubestates[k])][0][2]=1;
				
			}
			else if(i==4)
			{
				samplesTestAccessor[mark+k][6+GetFaceColor(12,cubestates[k])][0][0]=1;
				samplesTestAccessor[mark+k][12+GetFaceColor(13,cubestates[k])][2][0]=1;
				samplesTestAccessor[mark+k][24+GetFaceColor(14,cubestates[k])][2][2]=1;
				
			}
			else if(i==5)
			{
				samplesTestAccessor[mark+k][6+GetFaceColor(15,cubestates[k])][0][2]=1;
				samplesTestAccessor[mark+k][12+GetFaceColor(17,cubestates[k])][2][2]=1;
				samplesTestAccessor[mark+k][30+GetFaceColor(16,cubestates[k])][2][0]=1;
				
			}
			else if(i==6)
			{
				samplesTestAccessor[mark+k][6+GetFaceColor(18,cubestates[k])][2][2]=1;
				samplesTestAccessor[mark+k][30+GetFaceColor(20,cubestates[k])][2][2]=1;
				samplesTestAccessor[mark+k][18+GetFaceColor(19,cubestates[k])][2][0]=1;
				
			}
			else
			{
				samplesTestAccessor[mark+k][6+GetFaceColor(21,cubestates[k])][2][0]=1;
				samplesTestAccessor[mark+k][24+GetFaceColor(22,cubestates[k])][2][0]=1;
				samplesTestAccessor[mark+k][18+GetFaceColor(23,cubestates[k])][2][2]=1;			
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
		timeout=3;

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

template <class state, class action>
int LargeBatch<state,action>::GetFaceColor(int face,state& s)
{
	uint8_t cube = s.corner.state[face/3]; 
    uint8_t rot =  s.corner.state[8+cube]; 
    uint8_t result= cube*3+(3+(face%3)-rot)%3;

	int thecolor=-1;
    if (result==0)
      thecolor=0;
    else if (result==1)
      thecolor=4;
    else if (result==2)
      thecolor=2;
    else if (result==3)
      thecolor=0;
    else if (result==4)
      thecolor=2;
    else if (result==5)
      thecolor=5;
    else if (result==6)
      thecolor=0;
    else if (result==7)
      thecolor=5;
    else if (result==8)
      thecolor=3;
    else if (result==9)
      thecolor=0;
    else if (result==10)
      thecolor=3;
    else if (result==11)
      thecolor=4;
    else if (result==12)
      thecolor=1;
    else if (result==13)
      thecolor=2;
    else if (result==14)
      thecolor=4;
    else if (result==15)
      thecolor=1;
    else if (result==16)
      thecolor=5;
    else if (result==17)
      thecolor=2;
    else if (result==18)
      thecolor=1;
    else if (result==19)
      thecolor=3;
    else if (result==20)
      thecolor=5;
    else if (result==21)
      thecolor=1;
    else if (result==22)
      thecolor=4;
    else if (result==23)
      thecolor=3;
    else
		throw logic_error("we cannot assign the color.");
          
    return thecolor;
}

#endif