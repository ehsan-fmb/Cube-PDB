#include "IDAStar.h"
#include "ParallelIDAStar.h"
#include "batchIDAStar.h"
#include "Instances.h"
#include <stdexcept>
#include "DelayedHeuristicAStar.h"
#include "BatchABuffer.h"
#include "Timer.h"
#include "torch_tensorrt/torch_tensorrt.h"

using namespace std;

void warmup_model(torch::jit::script::Module& model, int gpu_core, int batch_size, int input_channels, int height, int width, int num_warmup_iterations) 
{
    // Create a dummy input tensor
    auto dummy_input = torch::randn({batch_size, input_channels, height, width}).to(devices[gpu_core]);
	dummy_input=dummy_input.to(at::kHalf);

    torch::InferenceMode guard;
	cudaSetDevice(gpu_core);
    for (int i = 0; i < num_warmup_iterations; ++i) {
        auto output = model.forward({dummy_input});
    }
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

void Test(string method, int num, int steps)
{
	
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath;
	Timer timer;
	cube.SetPruneSuccessors(true);
	
	load NN heuristics and use fp16 precision
	torch::jit::script::Module module_1=load_model(0);
	torch::jit::script::Module module_2=load_model(1);
	module_1.eval();
	module_2.eval();
	module_1.to(at::kHalf);
	module_2.to(at::kHalf);

	// use TensorRT to improve the performance of the models
	std::vector<int64_t> input_shape = {largebatchsize+lengthEpsilon, channels, height, width};
	torch_tensorrt::Input input(input_shape);

	torch_tensorrt::torchscript::CompileSpec compile_settings1({input});
	torch_tensorrt::torchscript::CompileSpec compile_settings2({input});

	compile_settings1.enabled_precisions = {torch::kHalf};
	compile_settings2.enabled_precisions = {torch::kHalf};

	cudaSetDevice(0);
	auto trt_model_1 = torch_tensorrt::torchscript::compile(module_1, compile_settings1);
	cudaSetDevice(1);
	auto trt_model_2 = torch_tensorrt::torchscript::compile(module_2, compile_settings2);

	warmup_model(trt_model_1,0,largebatchsize+lengthEpsilon,channels,height,width,10);
	warmup_model(trt_model_2,1,largebatchsize+lengthEpsilon,channels,height,width,10);

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

	double totalTime=0;
	int totalExpansion = 0, totalGenerated = 0;

	for (int x = 0; x < num; x++)
	{
		GetRandomN(start,steps,x);
		
		if (method=="BatchIDA")
		{
			printf("-=-=-BPIDA*-=-=-\n");
			const auto numThreads = thread::hardware_concurrency()-2;
			BatchIDAStar<RubiksCube, RubiksState, RubiksAction> bida(numThreads);
			// bida.SetNNHeuristics(trt_model_1,trt_model_2);
			bida.SetHeuristic(&h);
			bida.InitializeList();	
			timer.StartTimer();
			bida.GetPath(&cube, start, goal, rubikPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", bida.GetNodesExpanded(), bida.GetNodesTouched());
			printf("Solution path length %lu\n", rubikPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());

			totalTime+=timer.GetElapsedTime();
			totalExpansion+=bida.GetNodesExpanded();
			totalGenerated+=bida.GetNodesTouched();
		}
		else if(method=="ParallelIDA")
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

			totalTime+=timer.GetElapsedTime();
			totalExpansion+=pida.GetNodesExpanded();
			totalGenerated+=pida.GetNodesTouched();

		}
		else if(method=="StandardIDA")
		{
			printf("-=-=-IDA*-=-=-\n");
			IDAStar<RubiksState, RubiksAction> ida;
			ida.SetHeuristic(&h);
			timer.StartTimer();
			ida.GetPath(&cube, start, goal, rubikPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", ida.GetNodesExpanded(), ida.GetNodesTouched());
			printf("Solution path length %lu\n", rubikPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());

			totalTime+=timer.GetElapsedTime();
			totalExpansion+=ida.GetNodesExpanded();
			totalGenerated+=ida.GetNodesTouched();
		}
		else if(method=="BatchA")
		{
			printf("-=-=-BatchA*-=-=-\n");
			CNNHeuristicLookupBuffer buf;
			DelayedHeuristicAStar<RubiksState, RubiksAction, RubiksCube, CNNHeuristicLookupBuffer> a1(1000);
			a1.SetReopenNodes(true);
			a1.SetHeuristic(&h);
			timer.StartTimer();
			a1.GetPath(&cube, start, goal, rubikPath);
			timer.EndTimer();
			printf("%llu nodes expanded; %llu generated\n", a1.GetNodesExpanded(), a1.GetNodesTouched());
			printf("Solution path length %lu\n", rubikPath.size());
			printf("%1.2f elapsed\n", timer.GetElapsedTime());

			totalTime+=timer.GetElapsedTime();
			totalExpansion+=a1.GetNodesExpanded();
			totalGenerated+=a1.GetNodesTouched();
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
	int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
	std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
	
	string method=argv[1];
	int numTestcases=std::stoi(argv[2]);
	int steps=std::stoi(argv[3]);
	Test(method,numTestcases,steps);

	return 0;
}