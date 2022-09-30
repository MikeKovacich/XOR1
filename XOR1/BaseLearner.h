#pragma once
#include "pch.h"
#include "Environment.h"

struct BaseLearner {

	Environment* pEnv;
	string mName;
	unsigned mID;

	param_t mParams;
	vector<string> mStateLabels;
	// ctor
	BaseLearner(string Name, unsigned ID, param_t Params, vector<string> StateLabels, Environment* Env) :
	mName(Name), mID(ID), mParams(Params), mStateLabels(StateLabels),
		pEnv(Env) {}

	virtual void Step(state_t &x, const state_t & ex, value_t t, value_t dt) = 0;
};