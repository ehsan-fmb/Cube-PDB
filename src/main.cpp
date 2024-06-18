#include <iostream>
#include "batchIDAStar.h"
#include "RubiksCube.h"
#include "TopSpin.h"
#include "MM.h"

using namespace std;
const int N = 16;
const int K = 4;

void Test(int first, int last)
{
	
	BatchIDAStar<TopSpin<N, K>, TopSpinState<N>, TopSpinAction> bida;
	MM<TopSpinState<N>, TopSpinAction, TopSpin<N, K>> mm;
	TopSpin<N, K> ts;
	TopSpinState<N> s;
	TopSpinState<N> g;
	std::vector<TopSpinAction> actionPath;
	ts.StoreGoal(g);
	ZeroHeuristic<TopSpinState<N>> z;
	
	int table[] = {52058078,116173544,208694125,131936966,141559500,133800745,194246206,50028346,167007978,207116816,163867037,119897198,201847476,210859515,117688410,121633885};
	int table2[] = {145008714,165971878,154717942,218927374,182772845,5808407,19155194,137438954,13143598,124513215,132635260,39667704,2462244,41006424,214146208,54305743};
	for (int count = first; count < last; count++)
	{
		printf("Seed: %d\n", table[count&0xF]^table2[(count>>4)&0xF]);
		srandom(table[count&0xF]^table2[(count>>4)&0xF]);
		for (int x = 0; x < 200; x++)
			ts.ApplyAction(s, random()%N);
		Timer timer;
		
			
		printf("-=-=-PIDA*-=-=-\n");
		ts.SetPruneSuccessors(true);
		timer.StartTimer();
		bida.GetPath(&ts, s, g, actionPath);
		timer.EndTimer();
		printf("%llu nodes expanded; %llu generated\n", bida.GetNodesExpanded(), bida.GetNodesTouched());
		printf("Solution path length %lu\n", actionPath.size());
		printf("%1.2f elapsed\n", timer.GetElapsedTime());
		ts.SetPruneSuccessors(false);

	}
	mm.PrintHDist();
	
}


int main()
{
	Test(0,10);
	return 0;
}
