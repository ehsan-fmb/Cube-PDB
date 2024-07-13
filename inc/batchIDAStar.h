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


const int largebatchsize=1024;
const int largetimeout=3;
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
	int GetFaceColor(int face,state s);
	void GetNNInput(state s,torch::Tensor& input);
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
	torch::Tensor outputs;
	vector<torch::jit::IValue> inputs;
	SharedQueue<int> q;
	LargeBatch largeBatch;
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
	const auto numThreads = thread::hardware_concurrency();
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
	
	// double h=double(GetSavedHCost(threadID,node_index));

	// To get pdb results
	double h=heuristic->HCost(currState, goal);

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
		
		w.nodeCount++;
		indexes.push_back(w.nodeCount);

		batchUnit unit;
		torch::Tensor input;
		unit.index=w.nodeCount;
		unit.workNumber=threadID;
		GetNNInput(currState,input);
		children.push_back(input);
		units.push_back(unit);
		
		env->UndoAction(currState, actions[x]);
	}
	largeBatch.Add(children,units);

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
	std::lock_guard<std::mutex> lock(HCostListLock);
	return SavedHCosts[ID*numNodesWork+index];
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetNNOutput(const torch::Tensor& samples, torch::Tensor& h_values)
{
	inputs.resize(0);
	inputs.push_back(samples);
	// cout<<"size of samples: "<<samples.sizes()<<endl;
	// inputs.push_back(torch::zeros({1000,36,3,3}).to(device));
	outputs= model.forward(inputs).toTensor();
	outputs=torch::softmax(outputs,1);
	h_values= torch::argmax(outputs,1);
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch()
{
	while (!stopfeeder)
	{
		torch::Tensor samples;
		vector<batchUnit> units;
		bool full=largeBatch.IsFull(samples,units);

		// if(!full)
		// 	continue;

		// {
		// 	std::lock_guard<std::mutex> lock(HCostListLock);
				
		// 	// get hcosts from nn and copy units from largebatch
		// 	torch::Tensor h_values;
		// 	GetNNOutput(samples.to(device),h_values);
		// 	h_values=h_values.to(torch::kCPU);
			
		// 	feedcounter++;
		// 	totalsize=totalsize+units.size();
		// 	for (size_t i = 0; i < units.size(); i++) 
		// 	{
		// 		batchUnit unit=units[i];
		// 		// SavedHCosts[unit.workNumber*numNodesWork+unit.index]=2;
		// 		SavedHCosts[unit.workNumber*numNodesWork+unit.index]=h_values[i].item<int>();
		// 	}
		// }

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
void BatchIDAStar<environment, state, action>::GetNNInput(state s,torch::Tensor& input)
{
	input = torch::zeros({36,3,3});

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