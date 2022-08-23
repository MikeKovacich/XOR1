#pragma once
#include "pch.h"

struct Evaluation {

	string Label;
	unsigned numStepsTraining;
	unsigned numTrialsTraining;
	unsigned numStepsTesting;
	unsigned numTrialsTesting;
	unsigned numStepsPerCase;
	unsigned numStepsPerTrialTraining;
	unsigned numStepsPerTrialTesting;
	value_t dt;

	// default ctor
	Evaluation() {
		numStepsTraining = 1;
		numTrialsTraining = 1;
		numStepsTesting = 1;
		numTrialsTesting = 1;
		numStepsPerCase = 1;
		numStepsPerTrialTraining = 1;
		numStepsPerTrialTesting = 1;
		dt = 1.0;
	}

	// ctor
	Evaluation(string Label_, unsigned numStepsTraining_, unsigned numTrialsTraining_,
		unsigned numStepsTesting_, unsigned numTrialsTesting_, unsigned numStepsPerCase_, value_t dt_) :
		Label(Label_),
		numStepsTraining(numStepsTraining_),
		numTrialsTraining(numTrialsTraining_),
		numStepsTesting(numStepsTesting_),
		numTrialsTesting(numTrialsTesting_),
		numStepsPerCase(numStepsPerCase_),
		dt(dt_){}

};

