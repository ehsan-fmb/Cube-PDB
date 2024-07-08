#ifndef LARGEBATCH_H
#define LARGEBATCH_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <deque>
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
	void Add(const vector<torch::Tensor>& children,int work,const vector<batchUnit>& tempunits);
	void IsFull();
	void Empty();
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
}


LargeBatch::~LargeBatch()
{
}

/* Adds item to queue, regardless of capacity.  */
void LargeBatch::Add(const vector<torch::Tensor>& children,int work,const vector<batchUnit>& tempunits)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (!processing);});
	
	samples.insert(samples.end(),children.begin(),children.end());
	units.insert(units.end(),tempunits.begin(),tempunits.end());
	
    if(samples.size()>=maxbatchsize)
    {	
		processing=true;
		Full.notify_all();
	}

}

void LargeBatch::IsFull()
{
	processing=false;
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::milliseconds(timeout), [this](){return samples.size()>=maxbatchsize;});
	processing=true;
}

void LargeBatch::Empty()
{
	lock.lock();
    samples=vector<torch::Tensor>();
    units=vector<batchUnit>();
	processing=false;
    lock.unlock();
    Full.notify_all();
}


#endif
