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
	LargeBatch(int size);
	~LargeBatch();
	void Add(torch::Tensor input,int work,int counter);
	void IsFull();
	void Empty();
	vector<torch::Tensor> samples;
    vector<batchUnit> units;
	
private:
	int maxbatchsize;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
};


LargeBatch::LargeBatch(int size)
:maxbatchsize(size)
{
}


LargeBatch::~LargeBatch()
{
}

/* Adds item to queue, regardless of capacity.  */
void LargeBatch::Add(torch::Tensor input,int work,int counter)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (samples.size()!=maxbatchsize);});
	samples.push_back(input);
    batchUnit unit;
    unit.index=counter;
    unit.workNumber=work;
    units.push_back(unit);

    if(samples.size()==maxbatchsize)
        Full.notify_all();
}


void LargeBatch::IsFull()
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return samples.size()==maxbatchsize;});
}

void LargeBatch::Empty()
{
	lock.lock();
    samples=vector<torch::Tensor>();
    units=vector<batchUnit>();
    lock.unlock();
    Full.notify_all();
}


#endif
