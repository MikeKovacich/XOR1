#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"

struct AgentOutput : public BaseNodeGroup, public Agent
{
	value_t mGamma = 0.99;  // decay rate for OUTPUT
	//vector<unsigned> startSample;
	enum State_IDX { ID_idx, INP_idx, X_idx, INPOUT_idx, SPIKE_idx, SPIKET_idx, OUTPUT_idx };

	// ctor
	AgentOutput(string Name, unsigned ID, unsigned NumStates) :
		BaseNodeGroup(Name, ID, NumStates)
	{
		mLabel.push_back("INPOUT");
		mLabel.push_back("SPIKE");
		mLabel.push_back("SPIKET");
		mLabel.push_back("OUTPUT");
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}
	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt);
};
void AgentOutput::InitState(unsigned idx, state_t &x) {
	// ID, INP, X, SPIKE, SPIKET
	QIFInitState(x);

	// OUTPUT
	x[6] = 1.0;
}
void AgentOutput::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, INP, X, SPIKE, SPIKET
	QIFResetState(x);

	// OUTPUT
	x[6] = 1.0;
}
void AgentOutput::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
	// ID, INP, X, SPIKE, SPIKET
	QIFStepState(x, t, dt);  // Quadratic Integrate and Fire

	// OUTPUT

	// unpack
	value_t spike = x[4];
	value_t out = x[6];

	//compute
	out = mGamma * out + spike;  // likelihood
	cout << "AgOut  t:  " << t << " idx:  " << idx << " lhk:  " << out << endl;
	 
	// pack
	x[6] = out;
}

