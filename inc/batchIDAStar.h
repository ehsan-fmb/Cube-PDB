#ifndef BATCHIDASTAR_H
#define BATCHIDASTAR_H

#include "SearchEnvironment.h"
#include "FPUtil.h"
#include "vectorCache.h"
#include "SharedQueue.h"
#include "LargeBatch.h"
#include <thread>
#include <stack>
#include <inttypes.h>
#include <stdexcept>
#include <cassert>


constexpr int largebatchsize=8000;
constexpr int largetimeout=12000;
constexpr int numNodesWork=50000;
constexpr int stackNum=100;
constexpr int stackChunk=50;
constexpr int maxChildrenNum=20;

template <class state,class action>
struct StackUnit {
	state node;
	int index;
	action last;
	double gcost;

	// Default constructor
    StackUnit() {
    }

    StackUnit(StackUnit&& other) noexcept
        : node(std::move(other.node)), 
          index(other.index),
          last(other.last),
          gcost(other.gcost) {
    }

    StackUnit& operator=(StackUnit&& other) noexcept {
        if (this != &other) {
            node = std::move(other.node);
            index = other.index;
            last = other.last;
            gcost = other.gcost;
        }
        return *this;
    }
};

template <class environment, class state, class action>
class BatchIDAStar {
public:
	BatchIDAStar(int nT, string name);
	virtual ~BatchIDAStar() {}
	void GetPath(environment *env, state from, state to,
				 vector<action> &thePath);
	
	uint64_t GetNodesExpanded() { return nodesExpanded; }
	uint64_t GetNodesTouched() { return nodesTouched; }
	void ResetNodeCount() { nodesExpanded = nodesTouched = 0; }
	void SetNNHeuristics(torch::jit::script::Module& module1,torch::jit::script::Module& module2);
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}
	void InitializeList();
	double thecost;

private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound,
												int threadID,LargeBatch<state,action>& Batch);
	void AddWorkUnit(environment& env, StackUnit<state,action>& unit,BatchworkUnit<action>& localWork,double bound,int& nextValue,bool& nodeleft,int ID);
	bool DoIteration(environment *env, vector<StackUnit<state,action>>& nodes,double bound,
													BatchworkUnit<action> &w, vector<action> &actions,
													vector<state>& children,vector<int*>& indexes,vector<BatchworkUnit<action>*>& works);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	void UpdateNextBound(double currBound, double fCost);
	void GetNNOutput(torch::jit::script::Module& module,LargeBatch<state,action>& Batch,int n,torch::Tensor& outputs,torch::Tensor& tmp_hcosts);
	double GetSavedHCost(int ID,int index);
	void FeedLargeBatch(LargeBatch<state,action>& Batch,torch::jit::script::Module& module,torch::Tensor& outputs,torch::Tensor& tmp_hcosts);
	state goal,start;
	double nextBound;
	Heuristic<state> *heuristic;
	torch::jit::script::Module model1,model2;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<std::mutex> workLocks;
	mutable std::condition_variable workReady;
	vector<thread*> threads;
	vector<int>SavedHCosts;
	SharedQueue<int> q;
	LargeBatch<state,action> largeBatch1,largeBatch2;
	int foundSolution;
    bool finishAfterSolution,isRoot;
	bool stopfeeder;
	string env_name;
	int feedcounter,totalsize,numThreads;
	torch::Tensor outputs1,outputs2,tmp_hcosts1,tmp_hcosts2;
	MicroTimer timer;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar(int nT,string name):
finishAfterSolution(false),largeBatch1(largebatchsize,largetimeout,(nT*stackNum)/2,0),largeBatch2(largebatchsize,largetimeout,(nT*stackNum)/2,1),
	isRoot(true),workLocks(nT*stackNum),numThreads(nT),env_name(name)
{ 	
	outputs1= torch::empty({largebatchsize+lengthEpsilon,classNum}).to(devices[0]);
	outputs2= torch::empty({largebatchsize+lengthEpsilon,classNum}).to(devices[1]);

	tmp_hcosts1=torch::zeros({largebatchsize+lengthEpsilon},torch::dtype(torch::kInt64));
	tmp_hcosts2=torch::zeros({largebatchsize+lengthEpsilon},torch::dtype(torch::kInt64));
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::InitializeList()
{
	SavedHCosts.resize(numThreads*numNodesWork*stackNum,0);
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::SetNNHeuristics(torch::jit::script::Module& module1,torch::jit::script::Module& module2)
{
	model1=module1;
	model2=module2;
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GenerateWork(environment *env,
															   action forbiddenAction, state &currState,
															   vector<action> &thePath)
{

	if (thePath.size() >= batchWorkDepth)
	{
		BatchworkUnit<action> w;
		for (int x = 0; x < batchWorkDepth; x++)
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
		if(env_name=="RC")
		{
			if ((depth != 0) && (actions[x] == forbiddenAction))
			{
				nodesTouched--;
				continue;
			}	
		}
		
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
	start=from;
	
	if (env->GoalTest(from, to))
		return;
	
	vector<action> act;
	env->GetActions(from, act);
	
	double rootH =heuristic->HCost(from, to);
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
	thread largeBatchFeeder1(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this,std::ref(largeBatch1),std::ref(model1)
							,std::ref(outputs1),std::ref(tmp_hcosts1));
	thread largeBatchFeeder2(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this,std::ref(largeBatch2),std::ref(model2)
							,std::ref(outputs2),std::ref(tmp_hcosts2));
	feedcounter=0;
	totalsize=0;

	while (foundSolution > work.size())
	{
		
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);
		
		printf("Starting iteration with bound %f; %" PRId64 " expanded, %" PRId64 " generated\n", nextBound, nodesExpanded, nodesTouched);
		fflush(stdout); 
		
		for (size_t x = 0; x < work.size(); x++)
		{
			q.Add(x);
		}
		for (size_t x = 0; x < numThreads; x++)
		{
			if(x<numThreads/2)
			{
				threads.push_back(new thread(&BatchIDAStar<environment, state, action>::StartThreadedIteration, this,
												 *env, from, nextBound,x,std::ref(largeBatch1)));
			}
			else
			{
				threads.push_back(new thread(&BatchIDAStar<environment, state, action>::StartThreadedIteration, this,
												 *env, from, nextBound,x,std::ref(largeBatch2)));
			}
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
			largeBatchFeeder1.join();
			largeBatchFeeder2.join();
			return;
		}
	}
}


template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::AddWorkUnit(environment& env, StackUnit<state,action>& unit,BatchworkUnit<action>& localWork,
															double bound,int& nextValue,bool& nodeleft,int ID)
{
	// All values put in before threads start. Once the queue is empty we're done
	bool passedLimit = true;
	while(passedLimit)
	{

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
		
		passedLimit = false;
		for (int x = 0; x < batchWorkDepth; x++)
		{
			unit.gcost += env.GCost(unit.node, localWork.pre[x]);
			env.ApplyAction(unit.node, localWork.pre[x]);

			if (!passedLimit && fgreater(unit.gcost+heuristic->HCost(unit.node, goal), bound))
			{
				localWork.nextBound = unit.gcost+heuristic->HCost(unit.node, goal);
				passedLimit = true;
				unit.node=start;
				work[nextValue] = localWork;
				break;
			}

		}
		unit.last = localWork.pre[batchWorkDepth-1];
	}
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::StartThreadedIteration(environment env, state startState, double bound,
												int threadID,LargeBatch<state,action>& Batch)
{
		
	vector<action> actCache;
	vector<BatchworkUnit<action>*> workCache;
	vector<state> stateCache;
	vector<int*> indexCache;
	array<vector<StackUnit<state,action>>, stackNum> stacks;
	array<BatchworkUnit<action>,stackNum> threadworks;
	array<int,stackNum>nextvalues;
	array<int,stackNum>IDS;


	// allocate memory for vectors
	workCache.reserve(stackNum);
	stateCache.reserve(maxChildrenNum*stackNum);
	indexCache.reserve(maxChildrenNum*stackNum);

	// required parameters
	int counter=0;
	int miss=0;
	bool nodeLeft=true;
	
	// add initial workunits
	for (int i = 0; i < stackNum; i++) 
	{
		stacks[i].reserve(numNodesWork);
		
		StackUnit<state,action> unit;

		IDS[i]=threadID*stackNum+i;
		unit.index=0;
		unit.node=startState;

		AddWorkUnit(env,unit,threadworks[i],bound,nextvalues[i],nodeLeft,IDS[i]);
		
		if(!nodeLeft)
		{	
			IDS[i]=-1;
			miss++;
		}
		else
			stacks[i].push_back(std::move(unit));
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
				
				StackUnit<state,action> unit;
				unit.index=0;
				unit.node=startState;

				// get new workunit and break if there is no left
				AddWorkUnit(env,unit,threadworks[counter],bound,nextvalues[counter],nodeLeft,IDS[counter]);

				if(!nodeLeft)
				{
					miss++;
					IDS[counter]=-1;					
					break;
				}

				// set hcost of zero for root and push it to the stack
				stacks[counter].push_back(std::move(unit));	
			}

			costready=DoIteration(&env, stacks[counter], bound, threadworks[counter], actCache,stateCache,indexCache,workCache);
		}

		counter++;
		
		if(counter%stackChunk==0)
		{
			if(!indexCache.empty())
			{
				Batch.Add(stateCache,indexCache,workCache);
				stateCache.clear();
				indexCache.clear();
				workCache.clear();
			}
		}

		if(counter==stackNum)
			counter=0;	
		
	}

}

template <class environment, class state, class action>
bool BatchIDAStar<environment, state, action>::DoIteration(environment *env, vector<StackUnit<state,action>>& nodes,double bound,
															  BatchworkUnit<action> &w, vector<action> &actions,
															  vector<state>& children,vector<int*>& indexes,vector<BatchworkUnit<action>*>& works)
{

	// check if work is in process
	{
		std::lock_guard<std::mutex> lock(workLocks[w.ID]);
		if(w.processing)
			return false;	
	}	

	StackUnit<state,action> stackunit=std::move(nodes.back());
	state& currState=stackunit.node;
	int& node_index=stackunit.index;
	double& g=stackunit.gcost;
	action& forbiddenAction=stackunit.last;

	// double h=double(GetSavedHCost(w.ID,node_index));
	nodes.pop_back();

	// To get pdb results
	double h=heuristic->HCost(currState, goal);

	if (fgreater(g+h, bound))
	{
		if (g+h < w.nextBound)
			w.nextBound = g+h;
		return true;
	}

	// must do this after we check the f-cost bound
	if (env->GoalTest(currState, goal))
	{
		action a;
		w.solution.push_back(a);
		thecost=g;
		
		nodes.clear();

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
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (env_name=="RC")
		{
			if (actions[x] == forbiddenAction) 
			{
				nodesTouched--;
				continue;
			}
		}
				
		StackUnit<state,action> unit;
		
		unit.node=currState;

		double edgeCost = env->GCost(currState, actions[x]);
		env->ApplyAction(unit.node, actions[x]);
		w.nodeCount++;

		unit.index=w.nodeCount;
		unit.last=actions[x];
		unit.gcost=g+edgeCost;
		
		indexes.push_back(&SavedHCosts[w.ID*numNodesWork+w.nodeCount]);
		children.push_back(currState);
		nodes.push_back(std::move(unit));
		
		if (foundSolution <= w.unitNumber)
		{
			break;
		}
		
	}

	works.push_back(&w);
	return false;
}

template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int ID,int index)
{
	return SavedHCosts[ID*numNodesWork+index];
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetNNOutput(torch::jit::script::Module& module,LargeBatch<state,action>& Batch,int n,
												torch::Tensor& outputs,torch::Tensor& tmp_hcosts)
{
	torch::InferenceMode inference_mode;
	cudaSetDevice(Batch.num);
	
	Batch.narrow_cpu_tensor = Batch.samples.narrow(0, 0, n);
	Batch.gpu_slice = Batch.gpu_input.slice(0, 0, n);
	Batch.tmp_slice=tmp_hcosts.narrow(0,0,n);
	Batch.hcost_slice=Batch.h_values.slice(0,0,n);
	
	at::cuda::CUDAStreamGuard guard1(Batch.stream1);
	Batch.gpu_slice.copy_(Batch.narrow_cpu_tensor, true);

	at::cuda::CUDAStreamGuard guard2(Batch.stream2);
	outputs=module.forward({Batch.gpu_input}).toTensor();
	Batch.h_values= torch::argmax(outputs,1);

	// at::cuda::CUDAStreamGuard guard3(Batch.stream3);
	// Batch.tmp_slice.copy_(Batch.hcost_slice,true);

	Batch.stream1.synchronize();
	Batch.stream2.synchronize();
	// Batch.stream3.synchronize();
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch(LargeBatch<state,action>& Batch,torch::jit::script::Module& module,
							torch::Tensor& outputs,torch::Tensor& tmp_hcosts)
{
	while (!stopfeeder)
	{
		int wStart,uStart,wLength,uLength;
		bool full=Batch.IsFull(wStart,uStart,wLength,uLength);

		// cout<<"batch size: "<<uLength<<endl;

		if(!full)
			continue; 

		// get hcosts from nn
		GetNNOutput(module,Batch,uLength,outputs,tmp_hcosts);
		auto accessor= tmp_hcosts.accessor<long,1>();
		
		for (size_t i = 0; i <uLength; i++)
		{
			//units
			// *Batch.units[uStart+i]=accessor[i];
			
			//works
			if(i<wLength)
			{
				std::unique_lock<std::mutex> lock(workLocks[Batch.worksInProcess[wStart+i]->ID]);
				Batch.worksInProcess[wStart+i]->processing=false;	
			}
		
		}

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