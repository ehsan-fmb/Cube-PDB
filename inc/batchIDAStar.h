#ifndef BATCHIDASTAR_H
#define BATCHIDASTAR_H

#include "SearchEnvironment.h"
#include "FPUtil.h"
#include "vectorCache.h"
#include "SharedQueue.h"
#include "LargeBatch.h"
#include <thread>
#include <inttypes.h>
#include <stdexcept>
#include <cassert>
#include "Timer.h"


const int largebatchsize=10000;
const int largetimeout=2;
const int numNodesWork=30000;
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
	void SetNNHeuristic(torch::jit::script::Module& module){model=module;}
	void SetNNHeuristicTest(torch::jit::script::Module& module){model_test=module;}
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}

private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound,int threadID);
	void DoIteration(environment *env,
					 action forbiddenAction, state &currState,
					 vector<action> &thePath, double bound, double g,
					 BatchworkUnit<action> &w, vectorCache<action> &cache,int node_index,int threadID);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	void UpdateNextBound(double currBound, double fCost);
	void GetNNOutput(const torch::Tensor& samples,torch::Tensor& h_values);
	double GetSavedHCost(int ID,int index);
	void FeedLargeBatch();
	state goal;
	double nextBound;
	Heuristic<state> *heuristic;
	mutable std::mutex modelTestLock;
	mutable std::mutex HCostListLock;
	torch::jit::script::Module model;
	torch::jit::script::Module model_test;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<thread*> threads;
	vector<int>SavedHCosts;
	torch::Tensor outputs,tmp_hcosts;
	vector<torch::jit::IValue> inputs;
	SharedQueue<int> q;
	LargeBatch<state> largeBatch;
	int foundSolution;
    bool finishAfterSolution;
	bool stopfeeder;
	Timer timer;
	int feedcounter,totalsize;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar():
finishAfterSolution(false),largeBatch(largebatchsize,largetimeout)
{ 	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GenerateWork(environment *env,
															   action forbiddenAction, state &currState,
															   vector<action> &thePath)
{
		
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
	const auto numThreads = thread::hardware_concurrency()-1;
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
	
	double rootH = 11;
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
	thread largeBatchFeeder(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this);
	feedcounter=0;
	totalsize=0;
	SavedHCosts.resize(numThreads*numNodesWork,-1);

	while (foundSolution > work.size())
	{
		
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);
		
		timer.EndTimer();
		printf("%1.2f elapsed\n", timer.GetElapsedTime());
		cout<<"counter for feeder: "<<feedcounter<<endl;
		cout<<"totalsize of list: "<<totalsize<<endl;
		printf("Starting iteration with bound %f; %" PRId64 " expanded, %" PRId64 " generated\n", nextBound, nodesExpanded, nodesTouched);
		fflush(stdout);
		timer.StartTimer();
		
		for (size_t x = 0; x < work.size(); x++)
		{
			q.Add(x);
		}
		for (size_t x = 0; x < numThreads; x++)
		{
			threads.push_back(new thread(&BatchIDAStar<environment, state, action>::StartThreadedIteration, this,
												 *env, from, nextBound,x));
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
				return;
			}
	}
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::StartThreadedIteration(environment env, state startState, double bound,int threadID)
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
			DoIteration(&env, last, startState, thePath, bound, g, localWork, actCache,0,threadID);
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
															  BatchworkUnit<action> &w, vectorCache<action> &cache,int node_index, int threadID)
{
	
	double h=double(GetSavedHCost(threadID,node_index));

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
	vector<state>children;
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (actions[x] == forbiddenAction) 
			continue;
		
		env->ApplyAction(currState, actions[x]);
		
		w.nodeCount++;
		indexes.push_back(w.nodeCount);

		children.push_back(currState);
		
		env->UndoAction(currState, actions[x]);
	}
	largeBatch.Add(children,threadID,indexes);

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
		DoIteration(env, a,currState, thePath, bound, g+edgeCost, w, cache,indexes[j],threadID);
		j++;
		env->UndoAction(currState, actions[x]);
		thePath.pop_back();
		if (foundSolution <= w.unitNumber)
			break;
	}
	cache.returnItem(&actions);
}

template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int ID,int index)
{
	return SavedHCosts[ID*numNodesWork+index];
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetNNOutput(const torch::Tensor& samples, torch::Tensor& h_values)
{
	inputs.resize(0);
	inputs.push_back(samples);
	outputs= model.forward(inputs).toTensor();
	outputs=torch::softmax(outputs,1);
	h_values= torch::argmax(outputs,1);
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch()
{
	while (!stopfeeder)
	{
		
		bool full=largeBatch.IsFull();

		if(!full)
			continue;

		// get hcosts from nn and copy units from largebatch
		GetNNOutput(largeBatch.samples.to(device),largeBatch.h_values);
		tmp_hcosts=largeBatch.h_values.to(torch::kCPU);
		
		feedcounter++;
		totalsize=totalsize+largeBatch.units.size();
		for (size_t i = 0; i <5000; i++)
		{
			batchUnit& unit=largeBatch.units[i];
			SavedHCosts[unit.workNumber*numNodesWork+unit.index]=tmp_hcosts[i].item<int>();
		}

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

#endif