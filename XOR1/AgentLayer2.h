#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"

struct AgentLayer2 : public Agent
{
	enum State_IDX { ID_idx, INP_idx, X_idx, INPOUT_idx, SPIKE_idx, SPIKET_idx };
	//vector<unsigned> startSample;

	// ctor
	AgentLayer2(string Name, unsigned ID, unsigned NumStates) :
		Agent(Name, ID, NumStates)
	{
		mLabel.push_back("INPOUT");
		mLabel.push_back("SPIKE");
		mLabel.push_back("SPIKET");
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}
	//virtual void InitState(unsigned idx, state_t &x);
	//virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	//virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt);
};
//void AgentLayer2::InitState(unsigned idx, state_t &x) {
//	// ID, INP, X
//	QIFInitState(x);
//}
//void AgentLayer2::ResetState(unsigned idx, state_t &x, value_t* data) {
//	// ID, INP, X
//	QIFInitState(x);
//}
//void AgentLayer2::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
//	// ID, INP, X
//	QIFStepState(x, t, dt);  // Quadratic Integrate and Fire
//}

