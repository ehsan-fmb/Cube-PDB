#ifndef SMALLBATCH_H
#define SMALLBATCH_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;


class SmallBatch {
public:
	SmallBatch(int size,int t);
	~SmallBatch();
	int Add(const torch::Tensor& input);
	void IsFull();
	double GetHcost(int rank);
	void Inform(torch::Tensor &costs);
	vector<torch::Tensor> samples;	
	
private:
	bool costReady;
	bool processing;
	int maxbatchsize;
	int timeout;
	int counter;
	torch::Tensor hCosts;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
	mutable std::condition_variable ready;
};


SmallBatch::SmallBatch(int size,int t)
:maxbatchsize(size),costReady(false),timeout(t),processing(false)
{
}


SmallBatch::~SmallBatch()
{
}

/* Adds item to queue, regardless of capacity.  */
int SmallBatch::Add(const torch::Tensor& input)
{
	std::unique_lock<std::mutex> l(lock);
	ready.wait(l, [this](){return ((!costReady)&&(!processing));});

	samples.push_back(input);
	// notify batchfeeder if the batch is full
	if(samples.size()==maxbatchsize)
		Full.notify_one();

	return samples.size();
}

/* Adds item to queue, regardless of capacity.  */
double SmallBatch::GetHcost(int rank)
{
	std::unique_lock<std::mutex> l(lock);
	ready.wait(l, [this](){return costReady;});

	// Update the counter
	counter++;
    if (counter == hCosts.size(0))
	{
		costReady=false;
		ready.notify_all();
	}

	return hCosts[rank-1].item<double>();
}


void SmallBatch::IsFull()
{
	processing=false;
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::milliseconds(timeout), [this](){return samples.size()>=maxbatchsize;});
	processing=true;
}

void SmallBatch::Inform(torch::Tensor &costs)
{
	lock.lock();
	samples=vector<torch::Tensor>();
	hCosts=costs;
	counter=0;
	costReady=true;
	lock.unlock();
	ready.notify_all();
}

#endif
