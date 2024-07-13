#ifndef LARGEBATCH_H
#define LARGEBATCH_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <deque>
#include <memory>
#include <condition_variable>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;

struct batchUnit {
	int index;
	int workNumber;
};

class LargeBatch {
public:
	LargeBatch(int size,int t);
	~LargeBatch();
	void Add(vector<torch::Tensor>& children, vector<batchUnit>& tempunits);
	bool IsFull(torch::Tensor& states,vector<batchUnit>& nodes);
	vector<torch::Tensor> samples;
    vector<batchUnit> units;
	
private:
	int maxbatchsize;
	bool processing;
	int timeout;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
};


LargeBatch::LargeBatch(int size,int t)
:maxbatchsize(size),timeout(t),processing(false)
{
	samples.reserve(size);	
	units.reserve(size);
}


LargeBatch::~LargeBatch()
{
}

void LargeBatch::Add(vector<torch::Tensor>& children, vector<batchUnit>& tempunits)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return !(samples.size()>=maxbatchsize);});
	
	samples.insert(samples.end(),children.begin(),children.end());
	units.insert(units.end(),tempunits.begin(),tempunits.end());

    if(samples.size()>=maxbatchsize)
    {	
		Full.notify_all();
	}

}

bool LargeBatch::IsFull(torch::Tensor& states,vector<batchUnit>& nodes)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::milliseconds(timeout), [this](){return samples.size()>=maxbatchsize;});
	
	bool full=false;
	if(!samples.empty())
	{
		states=torch::stack(samples);
		nodes=units;
		full=true;
		samples.clear();
		units.clear();
	}
	
	Full.notify_all();
	return full;
}

#endif