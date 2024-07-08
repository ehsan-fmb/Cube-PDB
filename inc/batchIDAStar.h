#ifndef BATCHIDASTAR_H
#define BATCHIDASTAR_H

#include "SearchEnvironment.h"
#include "FPUtil.h"
#include "vectorCache.h"
#include "SharedQueue.h"
#include "SmallBatch.h"
#include "LargeBatch.h"
#include <thread>
#include <inttypes.h>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <cassert>
#include <memory>
#include "Timer.h"
#include <c10/cuda/CUDACachingAllocator.h>


const int smallbatchsize=30;
const int largebatchsize=3000;
const int smalltimeout=10;
const int largetimeout=50;
torch::Device device(torch::kCUDA,1);
using namespace std;

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
};

template <class environment, class state, class action>
class BatchIDAStar {
public:
	BatchIDAStar();
	virtual ~BatchIDAStar() {}
	void GetPath(environment *env, state from, state to,
				 vector<action> &thePath);
	
	uint64_t GetNodesExpanded() { return nodesExpanded; }
	uint64_t GetNodesTouched() { return nodesTouched; }
	void ResetNodeCount() { nodesExpanded = nodesTouched = 0; }
	void SetNNHeuristic(torch::jit::script::Module *module){model=module;}
	void SetNNHeuristicTest(torch::jit::script::Module *module){model_test=module;}
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}

private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound);
	void DoIteration(environment *env,
					 action forbiddenAction, state &currState,
					 vector<action> &thePath, double bound, double g,
					 BatchworkUnit<action> &w, vectorCache<action> &cache,int node_index);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	void UpdateNextBound(double currBound, double fCost);
	torch::Tensor GetNNOutput(const torch::Tensor& samples);
	double GetSavedHCost(int worknumber,int index,BatchworkUnit<action> &w);
	void FeedSmallBatch();
	void FeedLargeBatch();
	int GetFaceColor(int face,state s);
	void SetFrontiersHCost(environment *env);
	torch::Tensor GetNNInput(state s);
	state goal;
	double nextBound;
	Heuristic<state> *heuristic;
	mutable std::mutex modelLock;
	mutable std::mutex modelTestLock;
	mutable std::shared_mutex HCostListLock;
	torch::jit::script::Module *model;
	torch::jit::script::Module *model_test;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<thread*> threads;
	unordered_map<uint64_t, double> frontiers;
	vector<torch::jit::IValue> inputs;
	vector<torch::jit::IValue> inputs_test;
	std::vector<std::vector<double>> SavedHCosts;
	double frontiersmaxfcost;
	SharedQueue<int> q;
	SmallBatch smallBatch;
	LargeBatch largeBatch;
	int foundSolution;
    bool finishAfterSolution;
	bool stopfeeder;
	Timer timer;
	int feedcounter,totalsize,smallcounter,largecounter;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar():
finishAfterSolution(false),smallBatch(smallbatchsize,smalltimeout),largeBatch(largebatchsize,largetimeout)
{ 	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GenerateWork(environment *env,
															   action forbiddenAction, state &currState,
															   vector<action> &thePath)
{
	
	// add the state to the frontiers
	frontiers[env->GetStateHash(currState)]=0;
		
	if (thePath.size() >= workDepth)
	{
		BatchworkUnit<action> w;
		for (int x = 0; x < workDepth; x++)
		{
			w.pre[x] = thePath[x];
		}
		work.push_back(w);
		return;
	}
	
	vector<action> actions;
	env->GetActions(currState, actions);
	nodesTouched += actions.size();
	nodesExpanded++;
	int depth = (int)thePath.size();
	
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if ((depth != 0) && (actions[x] == forbiddenAction))
			continue;
		
		thePath.push_back(actions[x]);
		
		env->ApplyAction(currState, actions[x]);
			
		action a = actions[x];
		env->InvertAction(a);
		GenerateWork(env, a, currState, thePath);
		env->UndoAction(currState, actions[x]);
		thePath.pop_back();
	}
	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetPath(environment *env,
														  state from, state to,
														  vector<action> &thePath)
{
	
	const auto numThreads = thread::hardware_concurrency()-2;
	cout<<"number of threads: "<<numThreads<<endl;
	
	nextBound = 0;
	nodesExpanded = nodesTouched = 0;
	thePath.resize(0);
	work.resize(0);

	// Set class member
	goal = to;
	
	if (env->GoalTest(from, to))
		return;
	
	vector<action> act;
	env->GetActions(from, act);
	
	double rootH = 10;
	UpdateNextBound(0, rootH);
	
	// builds a list of all states at a fixed depth
	// we will then search them in parallel
	GenerateWork(env, act[0], from, thePath);
	for (size_t x = 0; x < work.size(); x++)
		work[x].unitNumber = x;
	printf("%lu pieces of work generated\n", work.size());
	foundSolution = work.size() + 1;

	// assign h-values for frontier nodes
	// SetFrontiersHCost(env);
	
	// define two threads that feed the nn with samll and large batches 
	stopfeeder=false;
	thread smallBatchFeeder(&BatchIDAStar<environment, state, action>::FeedSmallBatch, this);
	thread largeBatchFeeder(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this);
	feedcounter=0;
	totalsize=0;
	smallcounter=0;
	largecounter=0;

	while (foundSolution > work.size())
	{
		
		// initialize the SavedHCosts with number of works
		SavedHCosts.clear();
		SavedHCosts.resize(work.size());
		
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);
		
		timer.EndTimer();
		printf("%1.2f elapsed\n", timer.GetElapsedTime());
		cout<<"counter for feeder: "<<feedcounter<<endl;
		cout<<"totalsize of list: "<<totalsize<<endl;
		cout<<"query from large batch: "<<largecounter<<endl;
		cout<<"query from small batch: "<<smallcounter<<endl;
		printf("Starting iteration with bound %f; %" PRId64 " expanded, %" PRId64 " generated\n", nextBound, nodesExpanded, nodesTouched);
		fflush(stdout);
		timer.StartTimer();

		// erase frontiers if nextbound is greater than maximum fcost of frontiers
		// if (nextBound>frontiersmaxfcost && (! frontiers.empty()))
		// 	frontiers.clear();
		
		for (size_t x = 0; x < work.size(); x++)
		{
			q.Add(x);
		}
		for (size_t x = 0; x < numThreads; x++)
		{
			threads.push_back(new thread(&BatchIDAStar<environment, state, action>::StartThreadedIteration, this,
												 *env, from, nextBound));
		}
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x]->join();
			delete threads[x];
			threads[x] = 0;
		}
		double bestBound = (nextBound+1)*10; // FIXME: Better ways to do bounds
		for (int x = 0; x < work.size(); x++)
		{
			for (int y = 0; y < work[x].gHistogram.size(); y++)
			{
				gCostHistogram[y] += work[x].gHistogram[y];
				fCostHistogram[y] += work[x].fHistogram[y];
			}
			if (work[x].nextBound > nextBound && work[x].nextBound < bestBound)
			{
				bestBound = work[x].nextBound;
			}
			nodesExpanded += work[x].expanded;
			nodesTouched += work[x].touched;
			if (work[x].solution.size() != 0)
			{
				thePath = work[x].solution;
			}
		}
		nextBound = bestBound;
		if (thePath.size() != 0)
			{
				stopfeeder=true;
				largeBatchFeeder.join();
				smallBatchFeeder.join();
				return;
			}
	}
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::StartThreadedIteration(environment env, state startState, double bound)
{
		
	vectorCache<action> actCache;
	vector<action> thePath;
	while (true)
	{
		int nextValue;
		// All values put in before threads start. Once the queue is empty we're done
		if (q.Remove(nextValue) == false)
			break;
		
		thePath.resize(0);
		bool passedLimit = false;
		double g = 0;
		BatchworkUnit<action> localWork = work[nextValue];
		localWork.solution.resize(0);
		localWork.gHistogram.clear();
		localWork.gHistogram.resize(bound+1);
		localWork.fHistogram.clear();
		localWork.fHistogram.resize(bound+1);
		localWork.nextBound = 10*bound;//FIXME: Better ways to do this
		localWork.expanded = 0;
		localWork.touched = 0;
		localWork.nodeCount=0;

		for (int x = 0; x < workDepth; x++)
		{
			g += env.GCost(startState, localWork.pre[x]);
			env.ApplyAction(startState, localWork.pre[x]);
			thePath.push_back(localWork.pre[x]);

			// Check frontiers cost
			// if (bound<frontiersmaxfcost)
			// {
			// 	uint64_t hash=env.GetStateHash(startState);
			// 	if (!passedLimit && fgreater(g+frontiers[hash], bound))
			// 	{
			// 		localWork.nextBound = g+frontiers[hash];
			// 		passedLimit = true;
			// 	}
			// }

			// we instead check if the goal is in frontiers or not
			if(env.GoalTest(startState, goal))
			{
				cout<<"goal is in frontiers."<<endl;
				exit(-1);
			}

		}

		action last = localWork.pre[workDepth-1];
		env.InvertAction(last);
		
		if (!passedLimit)
		{
			DoIteration(&env, last, startState, thePath, bound, g, localWork, actCache,0);
		}
		
		for (int x = workDepth-1; x >= 0; x--)
		{
			env.UndoAction(startState, localWork.pre[x]);
			g -= env.GCost(startState, localWork.pre[x]);
		}
		work[nextValue] = localWork;

	}
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::DoIteration(environment *env,
															  action forbiddenAction, state &currState,
															  vector<action> &thePath, double bound, double g,
															  BatchworkUnit<action> &w, vectorCache<action> &cache,int node_index)
{
	
	double h=GetSavedHCost(w.unitNumber,node_index,w);
	if(h==-1)
	{
		torch::Tensor input=GetNNInput(currState);
		int index=smallBatch.Add(input);
		h = smallBatch.GetHcost(index);
		smallcounter++;
	}
	else
		largecounter++;
	
	// To get pdb results
	h=heuristic->HCost(currState, goal);

	if (fgreater(g+h, bound))
	{
		if (g+h < w.nextBound)
			w.nextBound = g+h;
		return;
	}

	// must do this after we check the f-cost bound
	if (env->GoalTest(currState, goal))
	{
		w.solution = thePath;
		foundSolution = min(foundSolution,w.unitNumber);
        if (finishAfterSolution)
            foundSolution = 0;
		return;
	}

	vector<action> &actions = *cache.getItem();
	
	env->GetActions(currState, actions);
	w.touched += actions.size();
	w.expanded++;
	w.gHistogram[g]++;
	w.fHistogram[g+h]++;

	// save nodes in large list to get their values when search
	// is back to upper levels
	vector<int> indexes;
	vector<torch::Tensor>children;
	vector<batchUnit>units;
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (actions[x] == forbiddenAction) 
			continue;
		
		env->ApplyAction(currState, actions[x]);

		indexes.push_back(w.nodeCount);
		batchUnit unit;
		unit.index=w.nodeCount;
		unit.workNumber=w.unitNumber;
		children.push_back(GetNNInput(currState));
		units.push_back(unit);
		
		env->UndoAction(currState, actions[x]);
		w.nodeCount++;
	}
	// add children to large batch
	largeBatch.Add(children,w.unitNumber,units);

	int j=0;
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (actions[x] == forbiddenAction) 
			continue;
		
		thePath.push_back(actions[x]);

		double edgeCost = env->GCost(currState, actions[x]);
		env->ApplyAction(currState, actions[x]);
		action a = actions[x];
		env->InvertAction(a);
		DoIteration(env, a, currState, thePath, bound, g+edgeCost, w, cache,indexes[j]);
		j++;
		env->UndoAction(currState, actions[x]);
		thePath.pop_back();
		if (foundSolution <= w.unitNumber)
			break;
	}
	cache.returnItem(&actions);
}


template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int worknumber,int index,BatchworkUnit<action> &w)
{
	std::shared_lock lock(HCostListLock);
	double cost= (index < SavedHCosts[worknumber].size()) ? SavedHCosts[worknumber][index] : -1;
	return cost;
}

template <class environment, class state, class action>
torch::Tensor BatchIDAStar<environment, state, action>::GetNNOutput(const torch::Tensor& samples)
{
	std::lock_guard<std::mutex> l(modelLock);

	// cout<<"smaples size: "<<samples.sizes()<<endl;
	inputs.push_back(samples);
	torch::Tensor outputs= model->forward(inputs).toTensor();
	torch::Tensor probs=torch::softmax(outputs,1);
	inputs=vector<torch::jit::IValue>();
	// c10::cuda::CUDACachingAllocator::emptyCache();
	return torch::argmax(probs,1);
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedSmallBatch()
{
	while (!stopfeeder)
	{
		smallBatch.IsFull();
		
		if(!smallBatch.samples.empty())
		{
			// cout<<"size of small batch: "<<batch.samples.size()<<endl;
			torch::Tensor h_values=GetNNOutput(torch::stack(smallBatch.samples));
			smallBatch.Inform(h_values);
		}
	}
	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch()
{
	while (!stopfeeder)
	{
		largeBatch.IsFull();

		if(largeBatch.samples.empty())
			continue;

		// put values in their corresponding position in the list
		torch::Tensor h_values=GetNNOutput(torch::stack(largeBatch.samples));
		// cout<<"size of large batch: "<<batch.samples.size()<<endl;
		{
			std::unique_lock lock(HCostListLock);
			feedcounter++;
			totalsize=totalsize+largeBatch.samples.size();
			for (size_t i = 0; i < largeBatch.units.size(); i++) 
			{
				batchUnit unit=largeBatch.units[i];
				SavedHCosts[unit.workNumber].push_back(h_values[i].item<double>());
			}
		}
		
		// clear the batch
		largeBatch.Empty();
	}
	
}


template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::UpdateNextBound(double currBound, double fCost)
{
	if (!fgreater(nextBound, currBound))
	{
		nextBound = fCost;
	}
	else if (fgreater(fCost, currBound) && fless(fCost, nextBound))
	{
		nextBound = fCost;
	}
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::SetFrontiersHCost(environment *env)
{
	double maxhcost=-1;
	vector<torch::Tensor> samples;
	vector<double*> values;
	const int length=80000;

	// Get NN output for frontiers
	for (auto it = frontiers.begin(); it != frontiers.end(); it++)
	{
		state s;
		auto next_it = std::next(it);
		env->GetStateFromHash(it->first,s);
		torch::Tensor input=GetNNInput(s);		
		samples.push_back(input);
		values.push_back(&it->second);

		if(samples.size()==length || next_it == frontiers.end())
		{
			torch::Tensor h_values = GetNNOutput(torch::stack(samples));
			for(unsigned int i = 0; i < samples.size(); i++)
			{
				*values[i]=h_values[i].item<double>();
				if (*values[i]>maxhcost)
					maxhcost=*values[i];		
			}
			samples=vector<torch::Tensor>();
			values=vector<double*>();
		}
	} 

	frontiersmaxfcost=workDepth+maxhcost;
}


template <class environment, class state, class action>
torch::Tensor BatchIDAStar<environment, state, action>::GetNNInput(state s)
{
	torch::Tensor input = torch::zeros({36,3,3});
	input=input.to(device);

	// // color center and edge cubies
	// for(int i = 0; i < 6; i++)
	// {
	// 	input[7*i][1][1]=1;
    //   	input[7*i][0][1]=input[7*i][1][0]=input[7*i][1][2]=input[7*i][2][1]=1;
	// }

	// // color corner cubies
	// for(int i = 0; i < 8; i++)
	// {
	// 	if(i==0)
	// 	{
	// 		input[GetFaceColor(0,s)][2][0]=1;
    //     	input[12+GetFaceColor(2,s)][0][0]=1;
    //     	input[24+GetFaceColor(1,s)][0][2]=1;
	// 	}
	// 	else if(i==1)
	// 	{
	// 		input[GetFaceColor(3,s)][2][2]=1;
    //     	input[12+GetFaceColor(4,s)][0][2]=1;
    //     	input[30+GetFaceColor(5,s)][0][0]=1;
	// 	}
	// 	else if(i==2)
	// 	{
	// 		input[GetFaceColor(6,s)][0][2]=1;
    //     	input[30+GetFaceColor(7,s)][0][2]=1;
    //     	input[18+GetFaceColor(8,s)][0][0]=1;

	// 	}
	// 	else if(i==3)
	// 	{
	// 		input[GetFaceColor(9,s)][0][0]=1;
    //     	input[24+GetFaceColor(11,s)][0][0]=1;
    //     	input[18+GetFaceColor(10,s)][0][2]=1;
			
	// 	}
	// 	else if(i==4)
	// 	{
	// 		input[6+GetFaceColor(12,s)][0][0]=1;
    //     	input[12+GetFaceColor(13,s)][2][0]=1;
    //     	input[24+GetFaceColor(14,s)][2][2]=1;
			
	// 	}
	// 	else if(i==5)
	// 	{
	// 		input[6+GetFaceColor(15,s)][0][2]=1;
    //     	input[12+GetFaceColor(17,s)][2][2]=1;
    //     	input[30+GetFaceColor(16,s)][2][0]=1;
			
	// 	}
	// 	else if(i==6)
	// 	{
	// 		input[6+GetFaceColor(18,s)][2][2]=1;
    //     	input[30+GetFaceColor(20,s)][2][2]=1;
    //     	input[18+GetFaceColor(19,s)][2][0]=1;
			
	// 	}
	// 	else
	// 	{
	// 		input[6+GetFaceColor(21,s)][2][0]=1;
    //     	input[24+GetFaceColor(22,s)][2][0]=1;
    //     	input[18+GetFaceColor(23,s)][2][2]=1;			
	// 	}
	// }

	return input;

}

template <class environment, class state, class action>
int BatchIDAStar<environment, state, action>::GetFaceColor(int face,state s)
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