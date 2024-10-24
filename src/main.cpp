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

torch::jit::script::Module load_model(int gpu_core)
{	
	
	//load the model
	torch::jit::script::Module module;
    try {
        module = torch::jit::load("../models/8-corners-2.4MB/model_traced.pt");
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

int GetFaceColor(int face,RubiksState s)
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

void GetNNInput(RubiksState s,torch::Tensor& input)
{
	input = torch::zeros({1,36,3,3},torch::dtype(at::kHalf));

	// color center and edge cubies
	for(int i = 0; i < 6; i++)
	{
		input[0][7*i][1][1]=1;
      	input[0][7*i][0][1]=input[0][7*i][1][0]=input[0][7*i][1][2]=input[0][7*i][2][1]=1;
	}

	// color corner cubies
	for(int i = 0; i < 8; i++)
	{
		if(i==0)
		{
			input[0][GetFaceColor(0,s)][2][0]=1;
        	input[0][12+GetFaceColor(2,s)][0][0]=1;
        	input[0][24+GetFaceColor(1,s)][0][2]=1;
		}
		else if(i==1)
		{
			input[0][GetFaceColor(3,s)][2][2]=1;
        	input[0][12+GetFaceColor(4,s)][0][2]=1;
        	input[0][30+GetFaceColor(5,s)][0][0]=1;
		}
		else if(i==2)
		{
			input[0][GetFaceColor(6,s)][0][2]=1;
        	input[0][30+GetFaceColor(7,s)][0][2]=1;
        	input[0][18+GetFaceColor(8,s)][0][0]=1;

		}
		else if(i==3)
		{
			input[0][GetFaceColor(9,s)][0][0]=1;
        	input[0][24+GetFaceColor(11,s)][0][0]=1;
        	input[0][18+GetFaceColor(10,s)][0][2]=1;
			
		}
		else if(i==4)
		{
			input[0][6+GetFaceColor(12,s)][0][0]=1;
        	input[0][12+GetFaceColor(13,s)][2][0]=1;
        	input[0][24+GetFaceColor(14,s)][2][2]=1;
			
		}
		else if(i==5)
		{
			input[0][6+GetFaceColor(15,s)][0][2]=1;
        	input[0][12+GetFaceColor(17,s)][2][2]=1;
        	input[0][30+GetFaceColor(16,s)][2][0]=1;
			
		}
		else if(i==6)
		{
			input[0][6+GetFaceColor(18,s)][2][2]=1;
        	input[0][30+GetFaceColor(20,s)][2][2]=1;
        	input[0][18+GetFaceColor(19,s)][2][0]=1;
			
		}
		else
		{
			input[0][6+GetFaceColor(21,s)][2][0]=1;
        	input[0][24+GetFaceColor(22,s)][2][0]=1;
        	input[0][18+GetFaceColor(23,s)][2][2]=1;			
		}
	}

}

void TestModel()
{
	
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath,a1;
	cube.SetPruneSuccessors(true);

	vector<int> blank;
	vector<int> corners;
	corners = {0, 1, 2, 3, 4, 5, 6, 7};
	RubikPDB pdb(&cube, goal, blank, corners);
	pdb.Load("../pdbs/8-corners/");
	Heuristic<RubiksState> h;
	h.lookups.push_back({kMaxNode, 1, 1});
	h.lookups.push_back({kLeafNode, 0, 0});
	h.heuristics.push_back(&pdb);


	vector<torch::jit::IValue> inputs;
	torch::jit::script::Module model=load_model(0);
	model.eval();
	model.to(at::kHalf);
	
	for (int t = 0; t < 1000; t++)
	{
		printf("Starting walk %d\n", t);
		for (int x = 0; x < 10000; x++)
		{
			cube.GetActions(start, a1);			
			rubikPath.push_back(a1[random()%a1.size()]);
			cube.ApplyAction(start, rubikPath.back());

			// Test the model
			double h_pdb=h.HCost(start, goal);
			
			torch::Tensor sample,outputs,h_values;
			GetNNInput(start,sample);
			sample=sample.to(devices[0]);
			inputs.resize(0);
			inputs.push_back(sample);
			outputs= model.forward(inputs).toTensor();
			h_values= torch::argmax(outputs,1);

			if(h_values[0].item<double>()==h_pdb)
			{
				// cout<<"PDB: "<<h_pdb<<endl;
				// cout<<"model: "<<h_values[0]<<endl;
				// cout<<"****************************"<<endl;
			}
			else
			{
				cout<<"PDB: "<<h_pdb<<endl;
				cout<<"model: "<<h_values[0]<<endl;
				cout<<"****************************"<<endl;
			}

		}
		while (rubikPath.size() > 0)
		{
			cube.UndoAction(start, rubikPath.back());
			rubikPath.pop_back();
		}
	}
	printf("Completed\n");
}


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

void Test(string method, int num, int steps)
{
	
	// rubik's cube environment
	RubiksCube cube;
	RubiksState start, goal;
	goal.Reset();
	std::vector<RubiksAction> rubikPath;
	Timer timer;
	cube.SetPruneSuccessors(true);
	
	// load NN heuristics and use fp16 precision
	torch::jit::script::Module module_1=load_model(0);
	torch::jit::script::Module module_2=load_model(1);
	module_1.eval();
	module_2.eval();
	module_1.to(at::kHalf);
	module_2.to(at::kHalf);

	warmup_model(module_1,0,largebatchsize+lengthEpsilon,channels,height,width,10);
	warmup_model(module_2,1,largebatchsize+lengthEpsilon,channels,height,width,10);

	// load 8-corners pdb heuristic
	vector<int> blank;
	vector<int> corners;
	// corners = {0, 1, 2, 3, 4, 5};
	// RubikPDB pdb(&cube, goal, blank, corners);
	// pdb.Load("../pdbs/");
	// pdb.DivCompress(4,true);
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
			BatchIDAStar<RubiksCube, RubiksState, RubiksAction> bida(numThreads,"RC");
			bida.SetNNHeuristics(module_1,module_2);
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
			printf("-=-=-StandardIDA*-=-=-\n");
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
	
	TestModel();
	// Test(method,numTestcases,steps);

	return 0;
}