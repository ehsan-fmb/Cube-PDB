#ifndef BATCHIDASTAR_H
#define BATCHIDASTAR_H

#include "SearchEnvironment.h"
#include "FPUtil.h"
#include "vectorCache.h"
#include "SharedQueue.h"
#include "LargeBatch.h"
#include "Timer.h"
#include <thread>
#include <stack>
#include <inttypes.h>
#include <stdexcept>
#include <cassert>


const int largebatchsize=4000;
const float largetimeout=1;
const int numNodesWork=50000;
const int stackNum=15;
torch::Device device(torch::kCUDA,1);
using namespace std;

template <class state>
struct StackUnit {
	state node;
	int index;
	int last;
	double gcost;
};

template <class environment, class state, class action>
class BatchIDAStar {
public:
	BatchIDAStar(int numThreads);
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
	void AddWorkUnit(environment& env, StackUnit<state>& unit,BatchworkUnit<action>& localWork,double& bound,int& nextValue,bool& nodeleft,int& ID);
	bool DoIteration(environment *env, stack<StackUnit<state>>& nodes,double& bound,
															  BatchworkUnit<action> &w, vectorCache<action> &cache);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	void UpdateNextBound(double currBound, double fCost);
	void GetNNOutput(const torch::Tensor& samples,torch::Tensor& h_values);
	double GetSavedHCost(int& ID,int& index);
	void FeedLargeBatch();
	state goal;
	double nextBound;
	Heuristic<state> *heuristic;
	mutable std::mutex modelTestLock;
	torch::jit::script::Module model;
	torch::jit::script::Module model_test;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<std::mutex> workLocks;
	mutable std::condition_variable workReady;
	vector<thread*> threads;
	vector<int>SavedHCosts;
	torch::Tensor outputs,tmp_hcosts;
	vector<torch::jit::IValue> inputs;
	SharedQueue<int> q;
	LargeBatch<state,action> largeBatch;
	int foundSolution;
    bool finishAfterSolution,isRoot;
	bool stopfeeder;
	Timer timer;
	int feedcounter,totalsize;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar(int numThreads):
finishAfterSolution(false),largeBatch(largebatchsize,largetimeout,numThreads*stackNum),isRoot(true),workLocks(numThreads*stackNum)
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
	if(isRoot)
	{
		isRoot=false;
		env->GetActions(currState, actions);
	}
	else
		env->GetActions(currState, actions,forbiddenAction);

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
	
	// define one thread that feeds the nn with large batches 
	stopfeeder=false;
	thread largeBatchFeeder(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this);
	feedcounter=0;
	totalsize=0;
	SavedHCosts.resize(numThreads*numNodesWork*stackNum,-1);

	while (foundSolution > work.size())
	{
		
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);

		//update timer
		largeBatch.UpdateTimer(largetimeout);
		
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
void BatchIDAStar<environment, state, action>::AddWorkUnit(environment& env, StackUnit<state>& unit,BatchworkUnit<action>& localWork,
															double& bound,int& nextValue,bool& nodeleft,int& ID)
{
	// All values put in before threads start. Once the queue is empty we're done
	if (q.Remove(nextValue) == false)
	{
		nodeleft= false;
		return;
	}
	else
		nodeleft= true;
	
	unit.gcost = 0;
	localWork = work[nextValue];
	localWork.solution.resize(0);
	localWork.gHistogram.clear();
	localWork.gHistogram.resize(bound+1);
	localWork.fHistogram.clear();
	localWork.fHistogram.resize(bound+1);
	localWork.nextBound = 10*bound;//FIXME: Better ways to do this
	localWork.expanded = 0;
	localWork.touched = 0;
	localWork.nodeCount=0;
	localWork.processing=false;
	localWork.ID=ID;

	for (int x = 0; x < workDepth; x++)
	{
		unit.gcost += env.GCost(unit.node, localWork.pre[x]);
		env.ApplyAction(unit.node, localWork.pre[x]);

		if(env.GoalTest(unit.node, goal))
		{
			cout<<"goal is in frontiers."<<endl;
			exit(-1);
		}

	}

	unit.last = localWork.pre[workDepth-1];
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::StartThreadedIteration(environment env, state startState, double bound,int threadID)
{
		
	vectorCache<action> actCache;
	array<stack<StackUnit<state>>, stackNum> stacks;
	array<BatchworkUnit<action>,stackNum> threadworks;
	array<int,stackNum>nextvalues;
	array<int,stackNum>IDS;

	// required parameters
	int counter=0;
	int miss=0;
	bool nodeLeft=true;
	
	// add initial workunits
	for (int i = 0; i < stackNum; i++) 
	{
		StackUnit<state> unit;
		IDS[i]=threadID*stackNum+i;
		unit.index=0;
		unit.node=startState;
		
		AddWorkUnit(env,unit,threadworks[i],bound,nextvalues[i],nodeLeft,IDS[i]);
		stacks[i].push(unit);
		SavedHCosts[IDS[i]*numNodesWork]=0;
    }

	while (miss<stackNum)
	{
		
		bool costready=true;
		while (costready)
		{
			
			if(stacks[counter].empty())
			{
				
				// restore initial paramters and save the work
				work[nextvalues[counter]] = threadworks[counter];

				StackUnit<state> unit;
				unit.index=0;
				unit.node=startState;

				// get new workunit and break if there is no left
				AddWorkUnit(env,unit,threadworks[counter],bound,nextvalues[counter],nodeLeft,IDS[counter]);
				
				if(!nodeLeft)
				{
					if(IDS[counter]!=-1)
					{
						miss++;
						IDS[counter]=-1;
					}					
					break;
				}

				// set hcost of zero for root and push it to the stack
				SavedHCosts[IDS[counter]*numNodesWork]=0;
				stacks[counter].push(unit);	
			}

			costready=DoIteration(&env, stacks[counter], bound, threadworks[counter], actCache);
		}

		counter++;
		if(counter==stackNum)
			counter=0;
	}

}

template <class environment, class state, class action>
bool BatchIDAStar<environment, state, action>::DoIteration(environment *env, stack<StackUnit<state>>& nodes,double& bound,
															  BatchworkUnit<action> &w, vectorCache<action> &cache)
{

	// check if work is in process
	{
		std::lock_guard<std::mutex> lock(workLocks[w.ID]);
		if(w.processing)
			return false;
	}	

	StackUnit<state> stackunit=nodes.top();
	state& currState=stackunit.node;
	int& node_index=stackunit.index;
	double& g=stackunit.gcost;
	action& forbiddenAction=stackunit.last;

	double h=double(GetSavedHCost(w.ID,node_index));
	nodes.pop();


	// To get pdb results
	h=heuristic->HCost(currState, goal);

	if (fgreater(g+h, bound))
	{
		if (g+h < w.nextBound)
			w.nextBound = g+h;
		return true;
	}

	// must do this after we check the f-cost bound
	if (env->GoalTest(currState, goal))
	{
		// w.solution = thePath;
		action a;
		w.solution.push_back(a);
		
		stack<StackUnit<state>> empty;
		nodes.swap(empty);

		foundSolution = min(foundSolution,w.unitNumber);
        if (finishAfterSolution)
            foundSolution = 0;
		return true;
	}

	vector<action> &actions = *cache.getItem();
	
	env->GetActions(currState, actions,forbiddenAction);
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
				
		StackUnit<state> unit;
		
		double edgeCost = env->GCost(currState, actions[x]);
		env->ApplyAction(currState, actions[x]);
		w.nodeCount++;

		unit.index=w.nodeCount;
		unit.node=currState;
		unit.last=actions[x];
		unit.gcost=g+edgeCost;
		
		indexes.push_back(w.nodeCount);
		children.push_back(currState);
		nodes.push(unit);
		
		env->UndoAction(currState, actions[x]);

		if (foundSolution <= w.unitNumber)
		{
			break;
		}
	}
	
	largeBatch.Add(children,indexes,&w);
	cache.returnItem(&actions);

	return true;
}

template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int& ID,int& index)
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

		// cout<<"size of samples:"<<largeBatch.mark<<endl;
		// get hcosts from nn and copy units from largebatch
		GetNNOutput(largeBatch.samples.to(device),largeBatch.h_values);
		tmp_hcosts=largeBatch.h_values.to(torch::kCPU);
		
		// --> This part is verified. It is commented because we should 
		// --> improve it. 
		// feedcounter++;
		// totalsize=totalsize+largeBatch.mark;
		// for (size_t i = 0; i <largeBatch.mark; i++)
		// {
		// 	batchUnit& unit=largeBatch.units[i];
		// 	SavedHCosts[unit.workNumber*numNodesWork+unit.index]=tmp_hcosts[i].item<int>();
		// }

		// --> This part is to enable works to continue
		for (size_t i = 0; i <largeBatch.worksInProcess.size(); i++)
		{
			std::unique_lock<std::mutex> lock(workLocks[largeBatch.worksInProcess[i]->ID]);
			largeBatch.worksInProcess[i]->processing=false;
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