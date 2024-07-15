#ifndef LARGEBATCH_H
#define LARGEBATCH_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <deque>
#include <memory>
#include <condition_variable>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;

struct batchUnit {
	int index;
	int workNumber;
};

template <class state>
class LargeBatch {
public:
	LargeBatch(int size,int t);
	~LargeBatch();
	void Add(vector<state>& cubestates, int& threadID, vector<int>& indexes);
	bool IsFull();
	void Empty();
	int GetFaceColor(int face,state s);
	void GetNNInput(state s,torch::Tensor& input);
	torch::Tensor samples,h_values;
	vector<batchUnit> units;

private:
	int maxbatchsize,mark;
	bool processing;
	int timeout;
	mutable std::mutex lock;
	mutable std::condition_variable Full;
};

template <class state>
LargeBatch<state>::LargeBatch(int size,int t)
:maxbatchsize(size),timeout(t),processing(false),samples(torch::zeros({10000, 7, 4, 4})),mark(0)
{	
	units.resize(size+100);
}

template <class state>
LargeBatch<state>::~LargeBatch()
{
}

template <class state>
void LargeBatch<state>::Add(vector<state>& cubestates, int& threadID, vector<int>& indexes)
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait(l, [this](){return (mark<maxbatchsize);});
	
	for(unsigned int i = 0; i < indexes.size(); i++)
  	{
    	// add new batchunits to units
		batchUnit unit;
		unit.index=indexes[i];
		unit.workNumber=threadID;
		units[mark+i]=unit;

		// edit samples with new states
  	}

	mark=mark+indexes.size();

    if(mark>=maxbatchsize)
    {	
		Full.notify_all();
	}

}


template <class state>
bool LargeBatch<state>::IsFull()
{
	std::unique_lock<std::mutex> l(lock);
	Full.wait_for(l, std::chrono::milliseconds(timeout), [this](){return mark>=maxbatchsize;});
	
	if(!units.empty())
		return true;
	else
		return false;

}

template <class state>
void LargeBatch<state>::Empty()
{
	lock.lock();
	mark=0;
	lock.unlock();
	Full.notify_all();
}

template <class state>
void LargeBatch<state>::GetNNInput(state s,torch::Tensor& input)
{
	input = torch::zeros({36,3,3});

	// // color center and edge cubies
	// for(int i = 0; i < 6; i++)
	// {
	// 	input[7*i][1][1]=1;
    //   	input[7*i][0][1]=input[7*i][1][0]=input[7*i][1][2]=input[7*i][2][1]=1;
	// }

	// // color corner cubies
	// for(int i = 0; i < 8; i++)
	// {
	// 	if(i==0)
	// 	{
	// 		input[GetFaceColor(0,s)][2][0]=1;
    //     	input[12+GetFaceColor(2,s)][0][0]=1;
    //     	input[24+GetFaceColor(1,s)][0][2]=1;
	// 	}
	// 	else if(i==1)
	// 	{
	// 		input[GetFaceColor(3,s)][2][2]=1;
    //     	input[12+GetFaceColor(4,s)][0][2]=1;
    //     	input[30+GetFaceColor(5,s)][0][0]=1;
	// 	}
	// 	else if(i==2)
	// 	{
	// 		input[GetFaceColor(6,s)][0][2]=1;
    //     	input[30+GetFaceColor(7,s)][0][2]=1;
    //     	input[18+GetFaceColor(8,s)][0][0]=1;

	// 	}
	// 	else if(i==3)
	// 	{
	// 		input[GetFaceColor(9,s)][0][0]=1;
    //     	input[24+GetFaceColor(11,s)][0][0]=1;
    //     	input[18+GetFaceColor(10,s)][0][2]=1;
			
	// 	}
	// 	else if(i==4)
	// 	{
	// 		input[6+GetFaceColor(12,s)][0][0]=1;
    //     	input[12+GetFaceColor(13,s)][2][0]=1;
    //     	input[24+GetFaceColor(14,s)][2][2]=1;
			
	// 	}
	// 	else if(i==5)
	// 	{
	// 		input[6+GetFaceColor(15,s)][0][2]=1;
    //     	input[12+GetFaceColor(17,s)][2][2]=1;
    //     	input[30+GetFaceColor(16,s)][2][0]=1;
			
	// 	}
	// 	else if(i==6)
	// 	{
	// 		input[6+GetFaceColor(18,s)][2][2]=1;
    //     	input[30+GetFaceColor(20,s)][2][2]=1;
    //     	input[18+GetFaceColor(19,s)][2][0]=1;
			
	// 	}
	// 	else
	// 	{
	// 		input[6+GetFaceColor(21,s)][2][0]=1;
    //     	input[24+GetFaceColor(22,s)][2][0]=1;
    //     	input[18+GetFaceColor(23,s)][2][2]=1;			
	// 	}
	// }

}

template <class state>
int LargeBatch<state>::GetFaceColor(int face,state s)
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