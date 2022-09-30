#pragma once
#include "pch.h"
#include "BaseEnvironment.h"
void unsignedToBinary(unsigned n, vector<bool> &y) {

	// counter for binary array
	int i = 0;
	while (n > 0) {

		// storing remainder in binary array
		y[i] = n % 2;
		n = n / 2;
		i++;
	}
}

struct Environment: public BaseEnvironment {

	state_t mReward;
	value_t mRewardTime;

	bool randomCases;
	Environment() : BaseEnvironment() {};
	Environment(string Label_, unsigned stepsPerCase_, unsigned numCase_, bool randomCases_) : 
		BaseEnvironment(Label_, stepsPerCase_, numCase_), randomCases(randomCases_){
		mReward.resize(3, 0.0);  // NOTE
	}
	bool booleanFunction(vector<bool> x);
	void Step_CPU(value_t t, value_t dt, vector<bool> &caseValue) {
		int nt = (int)(t / dt);
		div_t res = div(nt, stepsPerCaseBlock);
		int caseblk = res.rem;
		res = div(caseblk, stepsPerCase);
		int caseID = res.quot;
		unsignedToBinary(caseID, caseValue);
	}
	unsigned GetTrueAction(value_t t, value_t dt);
	void SetReward(state_t Reward, value_t Time) {
		mReward = Reward;
		mRewardTime = Time;
	}
	state_t GetReward(value_t time) { return mReward; }
	bool TimeToAct(value_t time, value_t dt);
};
bool Environment::TimeToAct(value_t time, value_t dt) {
	bool act = false;
	int nt = (int)(time / dt);
	div_t res = div(nt, stepsPerCaseBlock);
	int caseblk = res.rem;
	res = div(caseblk, stepsPerCase);
	if (res.rem == stepsPerCase - 1) act = true;  // last step in case
	return(act);
}

bool Environment::booleanFunction(vector<bool> x) {
	return (x[0] ^ x[1]);  // XOR
}

unsigned Environment::GetTrueAction(value_t t, value_t dt) {
	vector<bool> caseValue(2, 0);
	Step_CPU(t, dt, caseValue);
	unsigned trueAction = (unsigned) booleanFunction(caseValue);
	return(trueAction);
}
