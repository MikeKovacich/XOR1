#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"

struct AgentAction : public BaseNodeGroup
{
	enum State_IDX { ID_idx, INP_idx, X_idx, INPOUT_idx, NUMACTION_idx };
	unsigned mNumActions;
	//vector<value_t> startSample;

	// ctor
	AgentAction(string Name, unsigned ID, unsigned NumStates, unsigned NumActions) : 
		mNumActions(NumActions),
		BaseNodeGroup(Name, ID, NumStates)
	{
		mLabel.push_back("INPOUT");
		mLabel.push_back("NUMACTION");
		string actionLabel;
		for (unsigned i = 0; i < mNumActions; i++) {
			actionLabel = "ACTION" + to_string(i);
			mLabel.push_back(actionLabel);
		}
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}
	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt);
};
void AgentAction::InitState(unsigned idx, state_t &x) {
	// ID, INP, X, INPOUT, NUMACTION, ACTION1, ... , ACTIONN
	x[1] = 0.0;
	x[2] = 0.0;
	x[3] = 0.0;
	x[4] = mNumActions;
	for (unsigned i = 0; i < mNumActions; i++) {
		x[5 + i] = 1 / ((value_t) mNumActions);
	}
}
void AgentAction::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, INP, X, INPOUT, NUMACTION, ACTION1, ... , ACTIONN
	x[1] = 0.0;
	x[2] = 0.0;
	x[3] = 0.0;
	x[4] = mNumActions;
	for (unsigned i = 0; i < mNumActions; i++) {
		x[5 + i] = 1 / ((value_t)mNumActions);
	}
}
void AgentAction::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {
	// ID, INP, X, INPOUT, NUMACTION, ACTION1, ... , ACTIONN

	// unpack
	value_t I = x[1];
	value_t Action = x[2];
	value_t Iout = x[3];
	unsigned numAction = (unsigned) x[4];
	vector<value_t> prob(numAction);
	value_t sumProb = 0.0;
	for (unsigned i = 0; i < mNumActions; i++) {
		prob[i] = exp(x[5 + i]);
		sumProb = sumProb + prob[i];
	}
	for (unsigned i = 0; i < mNumActions; i++) {
		prob[i] = prob[i] / sumProb;
	}
	// local variables


	// compute
	Iout = I;
	discrete_distribution<int> distribution{ prob.begin(), prob.end() };
	Action = (value_t) distribution(rng);
	cout << "AgAct  t:  " << t << " idx:  " << idx << " prob:  " << prob[0] << ", " << prob[1] << " Action:  " << Action << endl;

	// pack
	x[1] = 0.0;
	x[2] = Action;
	x[3] = Iout;
}

