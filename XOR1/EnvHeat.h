#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"

struct EnvHeat : public BaseNodeGroup
{

	vector<unsigned> startSample;

	// ctor
	EnvHeat(string Name, unsigned ID, unsigned NumStates) :
		BaseNodeGroup(Name, ID, NumStates)
	{
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
		startSample.resize(mNumStates, 1);
	}
	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt);
};
void EnvHeat::InitState(unsigned idx, state_t &x) {
	// ID, INP, X
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvHeat::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, INP, X
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvHeat::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
	// ID, INP, X
	value_t INP = x[1];
	x[1] = 0.0;
	x[2] = x[2] + INP;
}

