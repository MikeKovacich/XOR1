#pragma once
#include "pch.h"

struct BaseEnvironment {

	string Label;
	unsigned stepsPerCase;
	unsigned numCase;
	unsigned stepsPerCaseBlock;

	BaseEnvironment() {
		Label = "Default";
		stepsPerCase = 500;
		numCase = 4;
		stepsPerCaseBlock = numCase * stepsPerCase;
	}
	BaseEnvironment(string Label_, unsigned stepsPerCase_, unsigned numCase_) : 
		Label(Label_),
		stepsPerCase(stepsPerCase_),
		numCase(numCase_) {
		stepsPerCaseBlock = numCase * stepsPerCase;
	};
	virtual bool booleanFunction(vector<bool> x) = 0;
	virtual void Step_CPU(value_t t, value_t dt, vector<bool> &caseValue) = 0;

};