#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"

struct EnvReward : public BaseNodeGroup
{
	enum State_IDX { ID_idx, ACT_idx, REW_idx, ACTOUT_idx };
	//vector<unsigned> startSample;
	Environment* pEnv;
	param_t mParams;

	// ctor
	EnvReward(string Name, unsigned ID, unsigned NumStates, Environment *Env, param_t Params) :

		BaseNodeGroup(Name, ID, NumStates), pEnv(Env), mParams(Params)
	{
		mLabel.push_back("ACTOUT");
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}
	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt);
};
void EnvReward::InitState(unsigned idx, state_t &x) {
	// ID, I = ACT, V = REW, ACTOUT
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvReward::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, I = ACT, V = REW, ACTOUT
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvReward::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
	// ID, I = ACT, V = REW, ACTOUT
	// unpack
	int agentAction = x[1];  // Action
	value_t gammaR = mParams["gammaR"];
	state_t reward = pEnv->GetReward(t);
	value_t R = reward[0];
	value_t Rbar = reward[1];
	value_t deltaR = reward[2];

	// local variables
	unsigned correctAction;

	// compute
	correctAction = pEnv->GetTrueAction(t, dt);
	R = 0.0;
	if (pEnv->TimeToAct(t, dt)) { 
		if (agentAction == correctAction) {
			R = 2.0;
		}
		else {
			R = -1.0;
		}
	}	
	deltaR = R - Rbar;
	Rbar = gammaR * Rbar + R;
	cout << "EnvRew  t:  " << t << " agentAction:  " << agentAction << 
								    " correctAction:  " << correctAction << 
		" R:  " << R << " Rbar:  " << Rbar << " deltaR:  " << deltaR << endl;
	// pack
	reward[0] = R;
	reward[1] = Rbar;
	reward[2] = deltaR;
	pEnv->SetReward(reward, t);
	x[1] = 0.0;
	x[2] = reward[0];
	x[3] = agentAction;
}

