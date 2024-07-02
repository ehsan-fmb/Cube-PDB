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
	void Add(torch::Tensor input,int work,int counter);
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
void LargeBatch::Add(torch::Tensor input,int work,int counter)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (!processing);});
	samples.push_back(input);
    batchUnit unit;
    unit.index=counter;
    unit.workNumber=work;
    units.push_back(unit);

    if(samples.size()==maxbatchsize)
    {	
		processing=true;
		Full.notify_all();
		cout<<"Large batch is full."<<endl;
	}
}

void LargeBatch::IsFull()
{
	processing=false;
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::milliseconds(timeout), [this](){return samples.size()==maxbatchsize;});
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
