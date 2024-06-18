#ifndef hog2_glut_ParallelIDA_h
#define hog2_glut_ParallelIDA_h

#include <iostream>
#include "SearchEnvironment.h"
#include <unordered_map>
#include "FPUtil.h"
#include "vectorCache.h"
#include "SharedQueue.h"
#include <thread>
#include <inttypes.h>

const int workDepth = 5;
using namespace std;

template <class action>
struct workUnit {
	action pre[workDepth];
	std::vector<action> solution;
	std::vector<int> gHistogram;
	std::vector<int> fHistogram;
	double nextBound;
	uint64_t expanded, touched;
	int unitNumber;
};

template <class environment, class state, class action>
class BatchIDAStar {
public:
	BatchIDAStar() { storedHeuristic = false; finishAfterSolution=false;}
	virtual ~BatchIDAStar() {}
	//	void GetPath(environment *env, state from, state to,
	//				 std::vector<state> &thePath);
	void GetPath(environment *env, state from, state to,
				 std::vector<action> &thePath);
	
	uint64_t GetNodesExpanded() { return nodesExpanded; }
	uint64_t GetNodesTouched() { return nodesTouched; }
	void ResetNodeCount() { nodesExpanded = nodesTouched = 0; }
	void SetHeuristic(Heuristic<state> *heur) { heuristic = heur; if (heur != 0) storedHeuristic = true;}
    void SetFinishAfterSolution(bool fas) { this->finishAfterSolution=fas;}
private:
	unsigned long long nodesExpanded, nodesTouched;
	
	void StartThreadedIteration(environment env, state startState, double bound);
	void DoIteration(environment *env,
					 action forbiddenAction, state &currState,
					 std::vector<action> &thePath, double bound, double g,
					 workUnit<action> &w, vectorCache<action> &cache);
	void GenerateWork(environment *env,
					  action forbiddenAction, state &currState,
					  std::vector<action> &thePath);
	
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
	void SetFrontiersHCost();
	state goal;
	double nextBound;
	bool storedHeuristic;
	Heuristic<state> *heuristic;
	std::vector<uint64_t> gCostHistogram;
	std::vector<uint64_t> fCostHistogram;
	std::vector<workUnit<action>> work;
	std::vector<std::thread*> threads;
	std::unordered_map<uint64_t, double> frontiers;
	double frontiersmaxfcost;
	SharedQueue<int> q;
	int foundSolution;
    bool finishAfterSolution;
};

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::GetPath(environment *env,
														  state from, state to,
														  std::vector<action> &thePath)
{
	const auto numThreads = std::thread::hardware_concurrency();

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
	
	std::vector<action> act;
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



	// assign h-values for frontier nodes
	SetFrontiersHCost();

	
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
			threads.push_back(new std::thread(&BatchIDAStar<environment, state, action>::StartThreadedIteration, this,
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
															   std::vector<action> &thePath)
{
	
	// add the state to the frontiers
	frontiers[env->GetStateHash(currState)]=0;
	
	
	if (thePath.size() >= workDepth)
	{
		workUnit<action> w;
		for (int x = 0; x < workDepth; x++)
		{
			w.pre[x] = thePath[x];
		}
		work.push_back(w);
		return;
	}
	
	std::vector<action> actions;
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
	std::vector<action> thePath;
	while (true)
	{
		int nextValue;
		// All values put in before threads start. Once the queue is empty we're done
		if (q.Remove(nextValue) == false)
			break;
		
		thePath.resize(0);
		bool passedLimit = false;
		double g = 0;
		workUnit<action> localWork = work[nextValue];
		localWork.solution.resize(0);
		localWork.gHistogram.clear();
		localWork.gHistogram.resize(bound+1);
		localWork.fHistogram.clear();
		localWork.fHistogram.resize(bound+1);
		localWork.nextBound = 10*bound;//FIXME: Better ways to do this
		localWork.expanded = 0;
		localWork.touched = 0;

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
	}
}


template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::DoIteration(environment *env,
															  action forbiddenAction, state &currState,
															  std::vector<action> &thePath, double bound, double g,
															  workUnit<action> &w, vectorCache<action> &cache)
{
	
	double h = heuristic->HCost(currState, goal);//, parentH); // TODO: restore code that uses parent h-cost
	
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
		foundSolution = std::min(foundSolution,w.unitNumber);
        if (finishAfterSolution)
            foundSolution = 0;
		return;
	}

	std::vector<action> &actions = *cache.getItem();
	
	env->GetActions(currState, actions);
	w.touched += actions.size();
	w.expanded++;
	w.gHistogram[g]++;
	w.fHistogram[g+h]++;

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

template <class environment, class state, class action>
void BatchIDAStar<environment, state, action>::SetFrontiersHCost()
{
	double maxhcost=100;
	// update the hcosts with NN and find the maximum hcost among them
	// for (auto& [hash, h_value] : frontiers) {
    // 	state s;
	// 	env->GetStateFromHash(s,hash);
	// 	h_value = random()%16;
	// }
	frontiersmaxfcost=workDepth+maxhcost;
}