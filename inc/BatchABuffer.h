#ifndef BATCHABUFFER_H
#define BATCHABUFFER_H


torch::jit::script::Module module_cnn_1_7, module_cnn_1_7_comp, module_cnn_8_15, module_cnn_8_15_comp;
torch::Device BatchAdevice = torch::Device(torch::kCUDA, 1);

void load_cnn() {
  at::globalContext().setBenchmarkCuDNN(true);
  try {
    module_cnn_1_7 = torch::jit::load("../models/model size experiment/moderate.pt", BatchAdevice);
    // module_cnn_1_7 = torch::jit::load("../models/STP44/cnn1-7c.pt", BatchAdevice);
    // module_cnn_1_7 = torch::jit::load("../models/model size experiment/heavy.pt", BatchAdevice);
    // module_cnn_1_7_comp  =torch::jit::load("../models/complement1-7q1c.pt", BatchAdevice);
    // module_cnn_8_15 = torch::jit::load("../models/cnn8-15c.pt", BatchAdevice);
    // module_cnn_8_15_comp  =torch::jit::load("../models/complement8-15c.pt", BatchAdevice);
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    // catch anything thrown within try block that derives from std::exception
    cerr << e.what();
  }
}

class CNNHeuristicLookupBuffer {
public:
	CNNHeuristicLookupBuffer()
	{
    nodeLimit = -1;
    load_cnn();
    
	}
	void Reset(MNPuzzle<4, 4> *env, const MNPuzzleState<4, 4> &goal, unsigned int nodeLimit)
	{
    this->env = env;
    results.resize(nodeLimit);
    index = 0;
    if (this->nodeLimit != nodeLimit) {
      this->nodeLimit = nodeLimit; // re-allocate memory for tensors
      tensor_1_7 = torch::zeros({nodeLimit, 7, 4, 4});

      // auto tensor_1_7_a = tensor_1_7.accessor<float,4>();
      // tensor_1_7_a[0][0][0][0] = 0;
		  tensor_8_15 = torch::zeros({nodeLimit, 8, 4, 4});
    } 
	}
	
	bool HitNodeLimit()
	{
		return index >= nodeLimit;
	}
	
	void Add(MNPuzzleState<4, 4> &puzzleState)
	{
    
    if (index == 0) {
      results.resize(nodeLimit);
      
      at::fill_(tensor_1_7, 0);
      at::fill_(tensor_8_15, 0);
    }
    
    auto tensor_1_7_a = tensor_1_7.accessor<float,4>();
    auto tensor_8_15_a = tensor_8_15.accessor<float,4>();
    int state_new[16];


    for (int j=0; j<16; j++) {
      state_new[puzzleState.puzzle[j]] = j;
    }

    for(int val=1; val<=7; val++) {
      int idx = state_new[val];
      tensor_1_7_a[index][val-1][idx/4][idx%4] = 1;
    }

    for(int val=8; val<=15; val++) {
      int idx = state_new[val];
      tensor_8_15_a[index][val-8][idx/4][idx%4] = 1;
    }

    // for (int j=0; j<16; j++) {
    //   cout<<puzzleState.puzzle[j]<<" ";
    // }
    // cout<<"*********************************"<<endl;


    
    // results[index] = env->HCost(puzzleState);
    results[index]=0;
    index++;
	}
	
	const vector<unsigned int> &Evaluate()
	{

    tensor_1_7_cuda = tensor_1_7.to(BatchAdevice);
    tensor_8_15_cuda = tensor_8_15.to(BatchAdevice); 
    get_heuristic_from_cnn_ens(tensor_1_7_cuda, module_cnn_1_7, 0, h_value_1_7);
    // get_heuristic_from_cnn_ens(tensor_1_7_cuda, module_cnn_1_7_comp, 0, complement_1_7);
    // get_heuristic_from_cnn_ens(tensor_8_15_cuda, module_cnn_8_15, 1, h_value_8_15);
    // get_heuristic_from_cnn_ens(tensor_8_15_cuda, module_cnn_8_15_comp, 1, complement_8_15);


    // t.StartTimer();
    // temp1 = heuristic_compliments.to(torch::kCPU);
    // temp2 = heuristic_values.to(torch::kCPU);
    // get_admissible_heuristics(complement_1_7, h_value_1_7, index, res_1_7);
    // get_admissible_heuristics(complement_8_15, h_value_8_15, index, res_8_15);
    // get_admissible_heuristics(complement_1_7, h_value_1_7, index, res_1_7);
    // get_admissible_heuristics(complement_8_15, h_value_8_15, index, res_8_15);
    // t.EndTimer();
    // total_get_admissive_h_time += t.GetElapsedTime();

    // t0.EndTimer();
    // total_eval_time += t0.GetElapsedTime();

    // for (int i=0; i<index; i++) {
    //   cout<<"heuristic 1: "<<res_1_7[i]<<endl;
    //   cout<<"heuristic 2: "<<res_8_15[i]<<endl;
    //   cout<<"*******************************"<<endl;
    // }  

    results.resize(index);
    for (int i=0; i<index; i++) {
      // results[i] += (res_1_7[i] + res_8_15[i]);
      results[i]+=1;
    }
    
    index = 0;
    return results;
    
	}
 
  void get_heuristic_from_cnn_ens(at::Tensor &tensor, torch::jit::script::Module &cnn, int type, at::Tensor &h_value) {
    // move it to cuda
    // temp = tensor.to(device);
    inputs.resize(0);
    inputs.push_back(tensor);

    // Execute the model and turn its output into a tensor.
    
    tensor_output = cnn.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    tensor_output = F::softmax(tensor_output, F::SoftmaxFuncOptions(1));
    h_value = at::argmax(tensor_output, 1);

  }

  void get_admissible_heuristics(at::Tensor &heuristic_compliments, at::Tensor &heuristic_values, int length, vector<int> &res) {
    res.resize(0);
    // move them back to cpu for faster access
    temp1 = heuristic_compliments.to(torch::kCPU);
    temp2 = heuristic_values.to(torch::kCPU);
    for (int i=0; i<length; i++) {
      if(temp2[i].item<int>() < temp1[i].item<int>()) {
        res.push_back(temp2[i].item<int>()*2);
      }
      else {
        res.push_back(temp1[i].item<int>()*2);
      }
    }
  } 
  
private:
	std::vector<unsigned int> results;
  int nodeLimit;
  at::Tensor tensor_1_7, tensor_8_15;
  at::Tensor h_value_1_7, complement_1_7, h_value_8_15, complement_8_15;
  unsigned int index;
  MNPuzzle<4, 4> *env;
  vector<int> res_1_7, res_8_15;
  at::Tensor tensor_output;
  std::vector<torch::jit::IValue> inputs;
  Timer t; // for timer
  at::Tensor temp, temp1, temp2, tensor_1_7_cuda, tensor_8_15_cuda;;

};

#endif