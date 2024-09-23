#include "IDAStar.h"
#include "ParallelIDAStar.h"
#include "batchIDAStar.h"
#include "DelayedHeuristicAStar.h"
#include "BatchABuffer.h"
#include <stdexcept>
#include "Timer.h"
#include "Instances.h"
#include "PermutationPDB.h"
#include "LexPermutationPDB.h"
#include "TemplateAStar.h"

using namespace std;


// Function to calculate total size of model parameters in bytes
size_t model_size_in_bytes(const torch::jit::script::Module& module) {
    size_t total_size = 0;
    
    // Iterate over all the named parameters in the TorchScript module
    for (const auto& param : module.named_parameters()) {
        const auto& tensor = param.value;  // Access the tensor directly
        total_size += tensor.numel() * tensor.element_size();  // numel() gives total elements, element_size() gives size in bytes per element
    }
    
    return total_size;
}

torch::jit::script::Module load_model(int gpu_core)
{	
	
	//load the model
	torch::jit::script::Module module;
    try {
        module = torch::jit::load("models/cnn1-7c.pt");
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

void Test(string method, int num, int steps)
{
	
	// rubik's cube environment
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath;
	Timer timer;
	cube.SetPruneSuccessors(true);

	// STP environment
	MNPuzzle<4, 4> mnp;
	MNPuzzleState<4, 4> s;
	MNPuzzleState<4, 4> g;
	std::vector<slideDir> stpPath;
	g.Reset();
	mnp.StoreGoal(g);

	
	// load NN heuristics and use fp16 precision
	torch::jit::script::Module module_1=load_model(0);
	torch::jit::script::Module module_2=load_model(1);
	module_1.eval();
	module_2.eval();
	module_1.to(at::kHalf);
	module_2.to(at::kHalf);

	size_t model_size = model_size_in_bytes(module_1);
	std::cout << "Model size in MB: " << model_size / (1024.0 * 1024.0) << " MB" << std::endl;


	// load 8-corners pdb heuristic
	// vector<int> blank;
	// vector<int> corners;
	// corners = {0, 1, 2, 3, 4, 5, 6, 7};
	// RubikPDB pdb(&cube, goal, blank, corners);
	// pdb.Load("../pdbs/8-corners/");
	// Heuristic<RubiksState> h;
	// h.lookups.push_back({kMaxNode, 1, 1});
	// h.lookups.push_back({kLeafNode, 0, 0});
	// h.heuristics.push_back(&pdb);

	// load 1-7 pdb heuristic
	std::vector<int> p1 = {0,1,2,3,4,5,6,7};
	std::vector<int> p2 = {0,8,9,10,11,12,13,14,15};
	STPLexPDB<4, 4> pdb1(&mnp, g, p1);
	STPLexPDB<4, 4> pdb2(&mnp, g, p2);
	mnp.SetPattern(p1);
	pdb1.Load("../pdbs/");
	mnp.SetPattern(p2);
	pdb2.Load("../pdbs/");
	Heuristic<MNPuzzleState<4, 4>> h;
	h.lookups.push_back({kAddNode, 1, 3});
	h.lookups.push_back({kLeafNode, 0, 0});
	h.lookups.push_back({kLeafNode, 1, 1});
	h.heuristics.push_back(&pdb1);
	h.heuristics.push_back(&pdb2);
	

	double totalTime=0;
	int totalExpansion = 0, totalGenerated = 0;

	for (int x = 0; x < num; x++)
	{
		GetRandomN(start,steps,x);
		// s=GetKorfInstance(x);
		
		if (method=="BatchIDA")
		{
			printf("-=-=-BPIDA*-=-=-\n");
			const auto numThreads = thread::hardware_concurrency()-1;
			// BatchIDAStar<RubiksCube, RubiksState, RubiksAction> bida(numThreads,"RC");
			BatchIDAStar<MNPuzzle<4, 4>, MNPuzzleState<4, 4>, slideDir> bida(numThreads,"STP");
			g.Reset();
			bida.SetNNHeuristics(module_1,module_2);
			bida.SetHeuristic(&h);
			bida.InitializeList();	
			timer.StartTimer();
			// bida.GetPath(&cube, start, goal, rubikPath);
			bida.GetPath(&mnp, s, g, stpPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", bida.GetNodesExpanded(), bida.GetNodesTouched());
			// printf("Solution path length %lu\n", rubikPath.size());
			printf("Solution path length %lu\n", stpPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());

			totalTime+=timer.GetElapsedTime();
			totalExpansion+=bida.GetNodesExpanded();
			totalGenerated+=bida.GetNodesTouched();
		}
		else if(method=="ParallelIDA")
		{	
			// printf("-=-=-PIDA*-=-=-\n");
			// // ParallelIDAStar<RubiksCube, RubiksState, RubiksAction> pida;
			// ParallelIDAStar<MNPuzzle<4, 4>, MNPuzzleState<4, 4>, slideDir> pida;
			// g.Reset();
			// pida.SetHeuristic(&h);
			// timer.StartTimer();
			// // pida.GetPath(&cube, start, goal, rubikPath);
			// pida.GetPath(&mnp, s, g, stpPath);
			// timer.EndTimer();
			// printf("%llu nodes expanded; %llu generated\n", pida.GetNodesExpanded(), pida.GetNodesTouched());
			// // printf("Solution path length %lu\n", rubikPath.size());
			// printf("Solution path length %lu\n", stpPath.size());
			// printf("%1.2f elapsed\n", timer.GetElapsedTime());

			// totalTime+=timer.GetElapsedTime();
			// totalExpansion+=pida.GetNodesExpanded();
			// totalGenerated+=pida.GetNodesTouched();

		}
		else if(method=="StandardIDA")
		{
			// printf("-=-=-StandardIDA*-=-=-\n");
			// // IDAStar<RubiksState, RubiksAction> ida;
			// IDAStar<MNPuzzleState<4, 4>, slideDir> ida;
			// g.Reset();
			// ida.SetHeuristic(&h);
			// timer.StartTimer();
			// // ida.GetPath(&cube, start, goal, rubikPath);
			// ida.GetPath(&mnp, s, g, stpPath);
			// timer.EndTimer();
			// printf("%llu nodes expanded; %llu generated\n", ida.GetNodesExpanded(), ida.GetNodesTouched());
			// printf("Solution path length %lu\n", stpPath.size());
			// // printf("Solution path length %lu\n", rubikPath.size());
			// printf("%1.2f elapsed\n", timer.GetElapsedTime());

			// totalTime+=timer.GetElapsedTime();
			// totalExpansion+=ida.GetNodesExpanded();
			// totalGenerated+=ida.GetNodesTouched();
		}
		else if(method=="BatchA")
		{
			// printf("-=-=-BatchA*-=-=-\n");
			// CNNHeuristicLookupBuffer buf;
			// DelayedHeuristicAStar<RubiksState, RubiksAction, RubiksCube, CNNHeuristicLookupBuffer> a1(1000);
			// // DelayedHeuristicAStar<MNPuzzleState<4, 4>, slideDir, MNPuzzle<4, 4>, CNNHeuristicLookupBuffer> a1(1000);
			// a1.SetReopenNodes(true);
			// a1.SetHeuristic(&h);
			// timer.StartTimer();
			// a1.GetPath(&cube, start, goal, rubikPath);
			// // a1.GetPath(&mnp, s, g, stpPath);
			// timer.EndTimer();
			// printf("%llu nodes expanded; %llu generated\n", a1.GetNodesExpanded(), a1.GetNodesTouched());
			// printf("Solution path length %lu\n", rubikPath.size());
			// // printf("Solution path length %lu\n", stpPath.size());
			// printf("%1.2f elapsed\n", timer.GetElapsedTime());

			// totalTime+=timer.GetElapsedTime();
			// totalExpansion+=a1.GetNodesExpanded();
			// totalGenerated+=a1.GetNodesTouched();
		}
		else if(method=="Template")
		{
			// printf("-=-=-TemplateA*-=-=-\n");
			// TemplateAStar<MNPuzzleState<4, 4>, slideDir, MNPuzzle<4, 4>> astar;
			// astar.SetHeuristic(&h);
			// timer.StartTimer();
			// astar.GetPath(&mnp, s, g, stpPath);
			// timer.EndTimer();
			// printf("%llu nodes expanded; %llu generated\n", astar.GetNodesExpanded(), astar.GetNodesTouched());
			// printf("Solution path length %lu\n", stpPath.size());
			// printf("%1.2f elapsed\n", timer.GetElapsedTime());

			// totalTime+=timer.GetElapsedTime();
			// totalExpansion+=astar.GetNodesExpanded();
			// totalGenerated+=astar.GetNodesTouched();

		}
		else
			throw invalid_argument( "method does not exist." );
		
	}

	cout<<"****************************************************\n";
	cout<<"****************************************************\n";
	cout<<"average time: "<<totalTime/num<<'\n';
	cout<<"average node expansion: "<<totalExpansion/num<<'\n';
	cout<<"average node generation: "<<totalGenerated/num<<'\n';
	
}


int main(int argc, char *argv[])
{	
	// int runtime_version = 0;
    // cudaRuntimeGetVersion(&runtime_version);
    // std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
	// std::cout << "LibTorch version: " << TORCH_VERSION;
	string method=argv[1];
	int numTestcases=std::stoi(argv[2]);
	int steps=std::stoi(argv[3]);
	Test(method,numTestcases,steps);

	return 0;
}