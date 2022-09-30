#pragma once
#include "pch.h"
#include "Environment.h"
#include "BaseNodeGroup.h"

struct EnvOutput : public BaseNodeGroup {

	//enum State_IDX { ID_idx, INP_idx, X_idx };
	Environment* pEnv;

	EnvOutput(string Name, unsigned ID, unsigned NumStates, Environment *Env) :
		BaseNodeGroup(Name, ID, NumStates), pEnv(Env) {
		//mLabel.push_back("X");
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	};
	virtual void InitState(unsigned indx, state_t &x);
	virtual void ResetState(unsigned indx, state_t &x, value_t* data);
	virtual void StepState(unsigned indx, state_t &x, value_t* data, value_t t, value_t dt);

};

void EnvOutput::InitState(unsigned indx, state_t &x) {
	// ID, INP, X
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvOutput::ResetState(unsigned indx, state_t &x, value_t* data) {
	// ID, INP, X
	x[1] = 0.0;
	x[2] = 0.0;
}
void EnvOutput::StepState(unsigned indx, state_t &x, value_t* data, value_t t, value_t dt) {

	// context
	vector<bool> caseValue(2, 0);
	pEnv->Step_CPU(t, dt, caseValue);


	// ID, INP, X
	x[1] = (value_t) caseValue[indx];
	x[2] = x[1];

	if (mName == "EnvOut") {
		cout << "StepState... EnvOut  t:  " << t << " indx:  " << indx << " ID:  " << x[0]
			<< " INP:  " << x[1] << " X:  " << x[2] << endl;
	}
}

