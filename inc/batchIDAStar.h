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


const int smallbatchsize=20;
const int largebatchsize=2000;
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
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur; if (heur != 0) storedHeuristic = true;}
	void SetNNHeuristic(torch::jit::script::Module *module){model=module;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}
private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound);
	void DoIteration(environment *env,
					 action forbiddenAction, state &currState,
					 vector<action> &thePath, double bound, double g,
					 BatchworkUnit<action> &w, vectorCache<action> &cache);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  vector<action> &thePath);
	
	void PrintGHistogram()
	{
		return;
		uint64_t early = 0, late = 0;
		printf("G-cost distribution\n");
		for (int x = 0; x < gCostHistogram.size(); x++)
		{
			if (gCostHistogram[x] != 0)
				printf("%d\t%" PRId64 "\n", x, gCostHistogram[x]);
			if (x*2 > gCostHistogram.size()-1)
				late += gCostHistogram[x];
			else
				early += gCostHistogram[x];
		}
		if (late < early)
			printf("--Strong heuristic - Expect MM > A*\n");
		else
			printf("--Weak heuristic - Expect MM >= MM0.\n");
		printf("\n");
		printf("F-cost distribution\n");
		for (int x = 0; x < fCostHistogram.size(); x++)
		{
			if (fCostHistogram[x] != 0)
				printf("%d\t%" PRId64 "\n", x, fCostHistogram[x]);
		}
		printf("\n");
	}
	void UpdateNextBound(double currBound, double fCost);
	torch::Tensor GetNNOutput(torch::Tensor samples);
	double GetSavedHCost(int worknumber,int index);
	void FeedSmallBatch(SmallBatch &batch);
	void FeedLargeBatch(LargeBatch &batch);
	int GetFaceColor(int face,state s);
	void SetFrontiersHCost(environment *env);
	torch::Tensor GetNNInput(state s);
	state goal;
	double nextBound;
	bool storedHeuristic;
	mutable std::mutex modelLock;
	mutable std::mutex HCostListLock;
	Heuristic<state> *heuristic;
	torch::jit::script::Module *model;
	vector<uint64_t> gCostHistogram;
	vector<uint64_t> fCostHistogram;
	vector<BatchworkUnit<action>> work;
	vector<thread*> threads;
	unordered_map<uint64_t, double> frontiers;
	vector<torch::jit::IValue> inputs;
	std::vector<std::vector<double>> SavedHCosts;
	double frontiersmaxfcost;
	SharedQueue<int> q;
	SmallBatch smallBatch;
	LargeBatch largeBatch;
	int foundSolution;
    bool finishAfterSolution;
};


template <class environment, class state, class action>
BatchIDAStar<environment, state, action>::BatchIDAStar():
storedHeuristic(false),finishAfterSolution(false),smallBatch(smallbatchsize),largeBatch(largebatchsize)
{ 	
}


template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetPath(environment *env,
														  state from, state to,
														  vector<action> &thePath)
{
	
	// define two threads that feed the nn with samll and large batches 
	thread smallBatchFeeder(&BatchIDAStar<environment, state, action>::FeedSmallBatch, this,ref(smallBatch));
	thread largeBatchFeeder(&BatchIDAStar<environment, state, action>::FeedLargeBatch, this,ref(largeBatch));
	
	const auto numThreads = thread::hardware_concurrency();
	cout<<"number of threads: "<<numThreads<<endl;

	if (!storedHeuristic)
		heuristic = env;
	
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
	
	double rootH = heuristic->HCost(from, to);
	UpdateNextBound(0, rootH);
	
	// builds a list of all states at a fixed depth
	// we will then search them in parallel
	GenerateWork(env, act[0], from, thePath);
	for (size_t x = 0; x < work.size(); x++)
		work[x].unitNumber = x;
	printf("%lu pieces of work generated\n", work.size());
	foundSolution = work.size() + 1;


	// initialize the SavedHCosts with number of works
	SavedHCosts.resize(work.size());



	// assign h-values for frontier nodes
	SetFrontiersHCost(env);

	
	while (foundSolution > work.size())
	{
		gCostHistogram.clear();
		gCostHistogram.resize(nextBound+1);
		fCostHistogram.clear();
		fCostHistogram.resize(nextBound+1);
		threads.resize(0);
		
		printf("Starting iteration with bound %f; %" PRId64 " expanded, %" PRId64 " generated\n", nextBound, nodesExpanded, nodesTouched);
		fflush(stdout);

		// erase frontiers if nextbound is greater than maximum fcost of frontiers
		if (nextBound>frontiersmaxfcost && (! frontiers.empty()))
			frontiers.clear();
		
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
			return;
	}
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
//		assert(!env->GoalTest(currState));
			
		action a = actions[x];
		env->InvertAction(a);
		GenerateWork(env, a, currState, thePath);
		env->UndoAction(currState, actions[x]);
		thePath.pop_back();
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

			if (bound<frontiersmaxfcost)
			{
				if (!passedLimit && fgreater(g+frontiers[env.GetStateHash(startState)], bound))
				{
					localWork.nextBound = g+frontiers[env.GetStateHash(startState)];
					passedLimit = true;
				}
			}
		}

		action last = localWork.pre[workDepth-1];
		env.InvertAction(last);
		
		if (!passedLimit)
		{
			DoIteration(&env, last, startState, thePath, bound, g, localWork, actCache);
		}
		
		for (int x = workDepth-1; x >= 0; x--)
		{
			env.UndoAction(startState, localWork.pre[x]);
			g -= env.GCost(startState, localWork.pre[x]);
		}
		work[nextValue] = localWork;


		// clear the work from SavedHCosts after its termination
		SavedHCosts[localWork.unitNumber]=std::vector<double>();

	}
}


template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::DoIteration(environment *env,
															  action forbiddenAction, state &currState,
															  vector<action> &thePath, double bound, double g,
															  BatchworkUnit<action> &w, vectorCache<action> &cache)
{
	
	double h=GetSavedHCost(w.unitNumber,w.nodeCount);
	if(h==-1)
	{
		torch::Tensor input=GetNNInput(currState);
		int index=smallBatch.Add(input);
		h = smallBatch.GetHcost(index);
	}

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
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (actions[x] == forbiddenAction) 
			continue;
		
		w.nodeCount++;
		env->ApplyAction(currState, actions[x]);
		largeBatch.Add(GetNNInput(currState),w.unitNumber,w.nodeCount);
		env->UndoAction(currState, actions[x]);
	}
	
	for (unsigned int x = 0; x < actions.size(); x++)
	{
		if (actions[x] == forbiddenAction) 
			continue;
		
		thePath.push_back(actions[x]);

		double edgeCost = env->GCost(currState, actions[x]);
		env->ApplyAction(currState, actions[x]);
		action a = actions[x];
		env->InvertAction(a);
		DoIteration(env, a, currState, thePath, bound, g+edgeCost, w, cache);
		env->UndoAction(currState, actions[x]);
		thePath.pop_back();
		if (foundSolution <= w.unitNumber)
			break;
	}
	cache.returnItem(&actions);
}


template <class environment, class state, class action>
double BatchIDAStar<environment, state, action>::GetSavedHCost(int worknumber,int index)
{
	std::lock_guard<std::mutex> l(HCostListLock);
	double cost= (index < SavedHCosts[worknumber].size()) ? SavedHCosts[worknumber][index] : -1;
	return cost;
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
	// update the hcosts with NN and find the maximum hcost among them
	for (auto it = frontiers.begin(); it != frontiers.end(); it++)
	{
    	state s;
		env->GetStateFromHash(it->first,s);
		torch::Tensor input=GetNNInput(s);		
		samples.push_back(input);
		values.push_back(&it->second);
		
		if(samples.size()==length)
		{
			torch::Tensor h_values = GetNNOutput(torch::stack(samples));
			for(unsigned int i = 0; i < length; i++)
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
torch::Tensor BatchIDAStar<environment, state, action>::GetNNOutput(torch::Tensor samples)
{
	std::lock_guard<std::mutex> l(modelLock);
	samples=samples.to(device);
	inputs.push_back(samples);
	torch::Tensor outputs= model->forward(inputs).toTensor();
	torch::Tensor probs=torch::softmax(outputs,1);
	inputs=vector<torch::jit::IValue>();
	return torch::argmax(probs,1);
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedSmallBatch(SmallBatch &batch)
{
	while (true)
	{
		batch.IsFull();
		torch::Tensor h_values=GetNNOutput(torch::stack(batch.samples));
		batch.Inform(h_values);
	}
	
}

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::FeedLargeBatch(LargeBatch &batch)
{
	while (true)
	{
		batch.IsFull();
		torch::Tensor h_values=GetNNOutput(torch::stack(batch.samples));

		// put values in their corresponding position in the list
		std::lock_guard<std::mutex> l(HCostListLock);
		for (size_t i = 0; i < batch.units.size(); ++i) 
		{
        	batchUnit unit=batch.units[i];
			SavedHCosts[unit.workNumber].push_back(h_values[i].item<double>());
			// debugging purpose
			assert(SavedHCosts[unit.workNumber].size()==unit.index && "The hcosts are not in the right order.");
    	}

		// clear the batch
		batch.Empty();
	}
	
}


template <class environment, class state, class action>
torch::Tensor BatchIDAStar<environment, state, action>::GetNNInput(state s)
{
	torch::Tensor input = torch::zeros({36,3,3});

	// color center and edge cubies
	for(int i = 0; i < 6; i++)
	{
		input[7*i][1][1]=1;
      	input[7*i][0][1]=input[7*i][1][0]=input[7*i][1][2]=input[7*i][2][1]=1;
	}

	// color corner cubies
	for(int i = 0; i < 8; i++)
	{
		if(i==0)
		{
			input[GetFaceColor(0,s)][2][0]=1;
        	input[12+GetFaceColor(2,s)][0][0]=1;
        	input[24+GetFaceColor(1,s)][0][2]=1;
		}
		else if(i==1)
		{
			input[GetFaceColor(3,s)][2][2]=1;
        	input[12+GetFaceColor(4,s)][0][2]=1;
        	input[30+GetFaceColor(5,s)][0][0]=1;
		}
		else if(i==2)
		{
			input[GetFaceColor(6,s)][0][2]=1;
        	input[30+GetFaceColor(7,s)][0][2]=1;
        	input[18+GetFaceColor(8,s)][0][0]=1;

		}
		else if(i==3)
		{
			input[GetFaceColor(9,s)][0][0]=1;
        	input[24+GetFaceColor(11,s)][0][0]=1;
        	input[18+GetFaceColor(10,s)][0][2]=1;
			
		}
		else if(i==4)
		{
			input[6+GetFaceColor(12,s)][0][0]=1;
        	input[12+GetFaceColor(13,s)][2][0]=1;
        	input[24+GetFaceColor(14,s)][2][2]=1;
			
		}
		else if(i==5)
		{
			input[6+GetFaceColor(15,s)][0][2]=1;
        	input[12+GetFaceColor(17,s)][2][2]=1;
        	input[30+GetFaceColor(16,s)][2][0]=1;
			
		}
		else if(i==6)
		{
			input[6+GetFaceColor(18,s)][2][2]=1;
        	input[30+GetFaceColor(20,s)][2][2]=1;
        	input[18+GetFaceColor(19,s)][2][0]=1;
			
		}
		else
		{
			input[6+GetFaceColor(21,s)][2][0]=1;
        	input[24+GetFaceColor(22,s)][2][0]=1;
        	input[18+GetFaceColor(23,s)][2][2]=1;			
		}
	}

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