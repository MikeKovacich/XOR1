#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"
#include "Agent.h"


struct AgentInput : public Agent
{

	enum State_IDX { ID_idx, INP_idx, X_idx, INPOUT_idx, SPIKE_idx, SPIKET_idx };

	//vector<unsigned> startSample;

	// ctor
	AgentInput(string Name, unsigned ID, unsigned NumStates) :
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
//void AgentInput::InitState(unsigned idx, state_t &x) {
//	// ID, INP, X, SPIKE, SPIKET
//
//	QIFResetState(x);
//
//	//// unpack
//	//value_t inputCurrent;
//	//value_t voltage;
//	//value_t spike;
//	//value_t spikeTime;
//
//	//// compute
//	//inputCurrent = 0.0;
//	//voltage = mVRest;
//	//spike = 0.0;
//	//spikeTime = -100.0;
//
//	//// pack
//	//x[1] = inputCurrent;
//	//x[2] = voltage;
//	//x[3] = spike;
//	//x[4] = spikeTime;
//}
//void AgentInput::ResetState(unsigned idx, state_t &x, value_t* data) {
//	// ID, INP, X, SPIKE, SPIKET
//
//	QIFResetState(x);
//
//	//// unpack
//	//value_t inputCurrent;
//	//value_t voltage;
//	//value_t spike;
//	//value_t spikeTime;
//
//	//// compute
//	//inputCurrent = 0.0;
//	//voltage = mVRest;
//	//spike = 0.0;
//	//spikeTime = -100.0;  // large time so exceeds refractory period
//
//	//// pack
//	//x[1] = inputCurrent;
//	//x[2] = voltage;
//	//x[3] = spike;
//	//x[4] = spikeTime;
//}
//void AgentInput::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
//
//	QIFStepState(x, t, dt);  // Quadratic Integrate and Fire
//
//}
//
