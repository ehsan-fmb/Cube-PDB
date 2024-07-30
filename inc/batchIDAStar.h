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
#include <cstdint>

const int largebatchsize=5000;
const int numNodesWork=50000;
const int stackNum=40;
const int maxChildrenNum=20;

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
	BatchIDAStar(int nT);
	virtual ~BatchIDAStar() {}
	void GetPath(environment *env, state from, state to,
				 vector<action> &thePath);
	
	uint64_t GetNodesExpanded() { return nodesExpanded; }
	uint64_t GetNodesTouched() { return nodesTouched; }
	void ResetNodeCount() { nodesExpanded = nodesTouched = 0; }
	void SetNNHeuristics(vector<torch::jit::script::Module>& modules){models=modules;}
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}
	void InitializeList();

private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound,int threadID);
	void AddWorkUnit(environment& env, StackUnit<state>& unit,BatchworkUnit<action>& localWork,double& bound,int& nextValue,bool& nodeleft,int& ID);
	bool DoIteration(environment *env, stack<StackUnit<state>>& nodes,double& bound,
															  BatchworkUnit<action> &w, vector<action> &actions,
															  vector<state> &children,vector<int*> &indexes,int& batch,
															  int& sNum, int& wNum);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	void UpdateNextBound(double currBound, double fCost);
	void GetNNOutput(int& index);
	double GetSavedHCost(int& ID,int& index);
	void FeedLargeBatch(int& batch,int& sNum, int& wNum);
	state goal;
	double nextBound;
	Heuristic<state> *heuristic;
	vector<torch::jit::script::Module> models;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<std::mutex> workLocks;
	mutable std::condition_variable workReady;
	vector<thread*> threads;
	vector<int>SavedHCosts;
	vector<torch::jit::IValue> inputs_0,inputs_1;
	SharedQueue<int> q;
	LargeBatch<state,action> largeBatch;
	int foundSolution;
    bool finishAfterSolution,isRoot;
	Timer timer;
	int feedcounter,totalsize,numThreads;
	vector<torch::Tensor> tmp_hcosts;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar(int nT):
finishAfterSolution(false),largeBatch(largebatchsize,nT*stackNum),isRoot(true),workLocks(nT*stackNum),numThreads(nT)
{ 	
	inputs_0.push_back(torch::jit::IValue());
	inputs_1.push_back(torch::jit::IValue());

	tmp_hcosts.resize(devices.size());
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::InitializeList()
{
	SavedHCosts.resize(numThreads*numNodesWork*stackNum,0);
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
	
	// batch process information. 
	feedcounter=0;
	totalsize=0;

	while (foundSolution > work.size())
	{
		
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);
		
		cout<<"counter for feeder: "<<feedcounter<<endl;
		cout<<"totalsize of list: "<<totalsize<<endl;
		printf("Starting iteration with bound %f; %" PRId64 " expanded, %" PRId64 " generated\n", nextBound, nodesExpanded, nodesTouched);
		fflush(stdout); 
		
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
	localWork.checked=0;

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
		
	vector<action> actCache;
	vector<state> stateCache(maxChildrenNum);
	vector<int*> indexCache(maxChildrenNum);
	array<stack<StackUnit<state>>, stackNum> stacks;
	array<BatchworkUnit<action>,stackNum> threadworks;
	array<int,stackNum>nextvalues;
	array<int,stackNum>IDS;
	int batch,sNum,wNum;

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
		stacks[i].push(move(unit));
    }

	while (miss<stackNum)
	{
		
		bool costready=true;
		while (costready)
		{
			
			if(IDS[counter]==-1)
				break;

			if(stacks[counter].empty())
			{
				
				// save the work
				work[nextvalues[counter]] = threadworks[counter];
				
				if(!nodeLeft)
				{
					miss++;
					IDS[counter]=-1;					
					break;
				}
				
				StackUnit<state> unit;
				unit.index=0;
				unit.node=startState;

				// get new workunit and break if there is no left
				AddWorkUnit(env,unit,threadworks[counter],bound,nextvalues[counter],nodeLeft,IDS[counter]);

				// set hcost of zero for root and push it to the stack
				stacks[counter].push(move(unit));	
			}

			costready=DoIteration(&env, stacks[counter], bound, threadworks[counter], actCache,stateCache,indexCache,batch,sNum,wNum);
		}

		counter++;
		if(counter==stackNum)
			counter=0;
	}

}

template <class environment, class state, class action>
bool BatchIDAStar<environment, state, action>::DoIteration(environment *env, stack<StackUnit<state>>& nodes,double& bound,
															  BatchworkUnit<action> &w, vector<action> &actions,
															  vector<state> &children,vector<int*> &indexes,int& batch,
															int& sNum, int& wNum)
{

	// check if work is in the process
	{
		std::unique_lock<std::mutex> lock(workLocks[w.ID]);
		if(w.processing)
		{
			if(w.checked)
			{
				w.checked++;
				if(w.checked>300 && largeBatch.GetStuck(w.b,sNum,wNum))
				{
					lock.unlock();
					FeedLargeBatch(w.b,sNum,wNum);
				}
				else
					return false;
			}
			else
				return false;
		}
			
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

	env->GetActions(currState, actions,forbiddenAction);
	w.touched += actions.size();
	w.expanded++;
	w.gHistogram[g]++;
	w.fHistogram[g+h]++;

	// save nodes in large list to get their values when search
	// is back to upper levels
	int j=0;
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
		
		indexes[j]=&SavedHCosts[w.ID*numNodesWork+w.nodeCount];
		children[j]=currState;
		nodes.push(move(unit));
		
		env->UndoAction(currState, actions[x]);

		j++;
		
		if (foundSolution <= w.unitNumber)
		{
			break;
		}
		
	}
	
	if(largeBatch.Add(children,indexes,&w,j,batch,sNum,wNum))
	{
		return false;
	}	
	else
	{
		FeedLargeBatch(batch,sNum,wNum);
		return true;
	}
		
}

template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int& ID,int& index)
{
	return SavedHCosts[ID*numNodesWork+index];
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetNNOutput(int& index)
{
	
	largeBatch.gpu_inputs[index].copy_(largeBatch.samples[index], true);
	
	if(index==0)
	{
		
		inputs_0[0]= torch::jit::IValue(std::ref(largeBatch.gpu_inputs[index]));
		largeBatch.outputs[index]= models[0].forward(inputs_0).toTensor();
	}
	else
	{
		inputs_1[0]= torch::jit::IValue(std::ref(largeBatch.gpu_inputs[index]));
		largeBatch.outputs[index]= models[1].forward(inputs_1).toTensor();
	}
	
	largeBatch.h_values[index]= torch::argmax(largeBatch.outputs[index],1);
	tmp_hcosts[index]=largeBatch.h_values[index].to(torch::kCPU);
	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch(int& batch,int& sNum, int& wNum)
{
	
	int wStart,uStart,wLength,uLength;

	wStart=batch*(numThreads*stackNum);
	uStart=batch*(lengthEpsilon+largebatchsize);
	wLength=wNum;
	uLength=sNum;

	// disable the work for checking the deadlock
	{
		std::unique_lock<std::mutex> lock(workLocks[largeBatch.worksInProcess[wStart]->ID]);
		largeBatch.worksInProcess[wStart]->checked=0;
	}

	largeBatch.Switch(batch);

	// get hcosts from nn 
	GetNNOutput(batch);

	auto accessor=tmp_hcosts[batch].accessor<long,1>();

	// cout<<"batch: "<<batch<<endl;
	// cout<<"size of batch: "<<uLength<<endl;
	// cout<<"***************************"<<endl;
	feedcounter++;
	totalsize=totalsize+uLength;
	for (size_t i = 0; i <uLength; i++)
	{
		
		// //units
		*largeBatch.units[uStart+i]=accessor[i];
	
		//works
		if(i<wLength)
		{
			std::unique_lock<std::mutex> lock(workLocks[largeBatch.worksInProcess[wStart+i]->ID]);
			largeBatch.worksInProcess[wStart+i]->processing=false;	
		}
	
	}

	largeBatch.Terminate(batch);
	
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