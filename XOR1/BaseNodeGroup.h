#pragma once
#include "pch.h"
#include "BaseNode.h"

struct BaseNodeGroup : public BaseNode
{
	
	enum State_IDX { ID_idx, INP_idx, X_idx };

	// ctor
	BaseNodeGroup(string Name, unsigned ID, unsigned NumStates) :
		BaseNode(Name, ID, NumStates)
	{
		mLabel.push_back("INP");
		mLabel.push_back("X");
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}

	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &xi, value_t* data, value_t t, value_t dt);

	//void Init();
	//void Reset(value_t* data);
	//void Step(value_t* data, value_t t, value_t dt);
};

void BaseNodeGroup::InitState(unsigned indx, state_t &x) {
	// ID, INP, X
	x[1] = 0.0;
	x[2] = 0.0;
}
void BaseNodeGroup::ResetState(unsigned indx, state_t &x, value_t* data) {
	// ID, INP, X
	x[INP_idx] = 0.0;
	x[X_idx] = 0.0;
}
void BaseNodeGroup::StepState(unsigned indx, state_t &x,  value_t* data, value_t t, value_t dt) {
	// ID, INP, X
	value_t INP = x[1];
	x[1] = 0.0;
	x[2] = x[2] + INP;
}


//__global__ void BaseNodeStep_GPU(value_t* NodeData, unsigned NodeLength,
//	value_t Drift, value_t Diffusion, value_t BoxWidth, value_t dt, curandState *states) {
//
//	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Constants
//	value_t minPos = 0.0;
//	value_t maxPos = BoxWidth;
//	// Unpack
//	unsigned idx = tid * NodeLength;
//	unsigned ID = (unsigned)NodeData[idx];
//	value_t X = NodeData[idx + 1];
//	value_t FX = NodeData[idx + 2];
//	unsigned CTR = (unsigned)NodeData[idx + 3];
//	value_t M = NodeData[idx + 4];
//
//	// Random Number Generator
//	CTR++;
//	curand_init(tid, CTR, 0, &states[tid]);
//
//	// Delta State
//	value_t dX;
//
//	dX = Drift * dt + FX * dt;
//
//	dX += Diffusion * sqrt(dt) * curand_normal(&states[tid]);
//
//	dX = dX / M;
//
//	// the dynamical equation 
//	X = X + dX;
//	if (X < minPos) { X = minPos; }
//	if (X > maxPos) { X = maxPos; }
//
//	// Pack
//	NodeData[idx + 1] = X;
//	NodeData[idx + 2] = 0.0;
//	NodeData[idx + 3] = CTR;
//
//	//__syncthreads();
//
//}
