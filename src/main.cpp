#include "IDAStar.h"
#include "ParallelIDAStar.h"
#include "batchIDAStar.h"
#include "singleIDAStar.h"
#include "RubiksCube.h"
#include <stdexcept>
#include "Timer.h"

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

	//print the state
	cout<<"*********************************"<<endl;
    printf("corners: %" PRIu64 "\n", start.corner.state);
	printf("edges: %" PRIu64 "\n", start.edge.state);
	cout<<"*********************************"<<endl;
}

torch::jit::script::Module load_model(int gpu_core)
{	
	
	//load the model
	torch::jit::script::Module module;
    try {
        module = torch::jit::load("../models/cnn1-7c.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
		exit(-1);
    }

	// Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Moving model to GPU." << std::endl;
        module.to(devices[gpu_core]);
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }

	return module;
}

void Test(string method)
{
	
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath;
	Timer timer;
	cube.SetPruneSuccessors(true);
	
	// load NN heuristics
	torch::jit::script::Module module_1=load_model(0);
	torch::jit::script::Module module_2=load_model(1);
	module_1.eval();
	module_2.eval();

	// load 8-corners pdb heuristic
	vector<int> blank;
	vector<int> corners;
	corners = {0, 1, 2, 3, 4, 5, 6, 7};
	RubikPDB pdb(&cube, goal, blank, corners);
	pdb.Load("../pdbs/8-corners/");
	
	Heuristic<RubiksState> h;
	h.lookups.push_back({kMaxNode, 1, 1});
	h.lookups.push_back({kLeafNode, 0, 0});
	h.heuristics.push_back(&pdb);


	const auto numThreads = thread::hardware_concurrency()-1;

	for (int x = 0; x < 1; x++)
	{
		GetRubikStep14Instance(start, x);
		
		if (method=="Batch")
		{
			printf("-=-=-BPIDA*-=-=-\n");
			BatchIDAStar<RubiksCube, RubiksState, RubiksAction> bida(numThreads);
			bida.SetNNHeuristics(module_1,module_2);
			bida.SetHeuristic(&h);
			bida.InitializeList();	
			timer.StartTimer();
			bida.GetPath(&cube, start, goal, rubikPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", bida.GetNodesExpanded(), bida.GetNodesTouched());
			printf("Solution path length %lu\n", rubikPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());
		}
		else if(method=="Parallel")
		{	
			printf("-=-=-PIDA*-=-=-\n");
			ParallelIDAStar<RubiksCube, RubiksState, RubiksAction> pida;
			pida.SetHeuristic(&h);
			timer.StartTimer();
			pida.GetPath(&cube, start, goal, rubikPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", pida.GetNodesExpanded(), pida.GetNodesTouched());
			printf("Solution path length %lu\n", rubikPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());

		}
		else if(method=="Single")
		{

		}
		else if(method=="Standard")
		{

		}
		else
			throw invalid_argument( "method does not exist." );

	}
	
}

int main(int argc, char *argv[])
{
	int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
	std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
	string method=argv[1];
	Test(method);
	return 0;
}
