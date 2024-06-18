#include <iostream>
#include "batchIDAStar.h"
#include "RubiksCube.h"
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;

void GetRubikStep14Instance(RubiksState &start, int which)
{
	RubiksCube c;
	int table[] = {52058078,116173544,208694125,131936966,141559500,133800745,194246206,50028346,167007978,207116816,163867037,119897198,201847476,210859515,117688410,121633885};
	int table2[] = {145008714,165971878,154717942,218927374,182772845,5808407,19155194,137438954,13143598,124513215,132635260,39667704,2462244,41006424,214146208,54305743};
	int first = 0, last = 50;
	srandom(table[which&0xF]^table2[(which>>4)&0xF]);
	
	start.Reset();
	for (int x = 0; x < 14; x++)
	{
		c.ApplyAction(start, random()%18);
	}
}

void Test()
{
	
	BatchIDAStar<RubiksCube, RubiksState, RubiksAction> bida;
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath;
	Timer timer;
	cube.SetPruneSuccessors(true);
	
	printf("-=-=-PIDA*-=-=-\n");
	for (int x = 0; x < 100; x++)
	{
		GetRubikStep14Instance(start, x);	
		goal.Reset();	
		timer.StartTimer();
		bida.GetPath(&cube, start, goal, rubikPath);
		timer.EndTimer();
		printf("%llu nodes expanded; %llu generated\n", bida.GetNodesExpanded(), bida.GetNodesTouched());
		printf("Solution path length %lu\n", rubikPath.size());
		printf("%1.2f elapsed\n", timer.GetElapsedTime());
	}
	
}


int main()
{
	Test();
	return 0;
}
