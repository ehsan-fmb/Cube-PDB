#ifndef INSTANCES_h
#define INSTANCES_h


#include <stdio.h>
#include "RubiksCube.h"
#include "RC.h"
#include "MNPuzzle.h"


void GetRandomN(RubiksState &start, int N, int which)
{
	start.Reset();
	RubiksCube c;
	srandom(which);
	std::vector<RubiksAction> acts;
	c.SetPruneSuccessors(true);
	for (int x = 0; x < N; x++)
	{
		c.GetActions(start, acts);
		c.ApplyAction(start, acts[random()%acts.size()]);
	}
}

MNPuzzleState<4, 4> GetRandomInstance(int walkLength)
{
    MNPuzzle<4, 4> p;
    MNPuzzleState<4, 4> s;
    std::vector<slideDir> acts;
    for (int x = 0; x < walkLength; x++)
    {
        p.GetActions(s, acts);
        p.ApplyAction(s, acts[random()%acts.size()]);
    }
    return s;
}

#endif