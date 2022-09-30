#pragma once
#include "BaseLearner.h"

struct Learner1 : public BaseLearner {

	Learner1(string Name, unsigned ID, param_t Params, vector<string> StateLabels, Environment* Env) :
		BaseLearner(Name, ID, Params, StateLabels, Env) {}

	void Step(state_t &x, const state_t & ex, value_t t, value_t dt);
};

void Learner1::Step(state_t &x, const state_t & ex, value_t t,  value_t dt) {

	// -----------------
	// unpack
	// -----------------
	state_t reward = pEnv->GetReward(t);
	value_t R = reward[0];
	value_t Rbar = reward[1];
	value_t deltaR = reward[2];
	value_t gammaEp = mParams["gammaEp"];
	value_t gammaEm = mParams["gammaEm"];
	value_t gammaSTDP = mParams["gammaSTDP"];
	value_t etaW = mParams["etaW"];
	value_t W = x[4];
	value_t eligPlus = x[6];
	value_t eligMinus = x[7];
	value_t stdp = x[8];
	value_t spikePred = ex[0];
	value_t spikeSucc = ex[1];

	// -----------------
	// local variables
	// -----------------	
	value_t deltaSTDP;
	if (spikePred > 0 || spikeSucc > 0) {
		int xx = 0;
	}

	// -----------------
	// compute
	// -----------------	
	eligPlus = (1 - spikePred) * gammaEp * eligPlus + spikePred;
	eligMinus = (1 - spikeSucc) * gammaEm * eligMinus + spikeSucc;
	deltaSTDP = eligPlus * spikeSucc - eligMinus * spikePred;
	stdp = gammaSTDP * stdp + deltaSTDP;

	if (pEnv->TimeToAct(t, dt)) {
		W = W + etaW * deltaR * stdp;
	}

	// -----------------
	// pack
	// -----------------	
	x[4] = W;
	x[6] = eligPlus;
	x[7] = eligMinus;
	x[8] = stdp;
}


