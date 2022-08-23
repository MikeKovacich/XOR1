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

	bool randomCases;
	Environment() : BaseEnvironment() {};
	Environment(string Label_, unsigned stepsPerCase_, unsigned numCase_, bool randomCases_) : 
		BaseEnvironment(Label_, stepsPerCase_, numCase_), randomCases(randomCases_){ }
	bool booleanFunction(vector<bool> x);
	void Step_CPU(value_t t, value_t dt, vector<bool> &caseValue) {
		int nt = (int)(t / dt);
		div_t res = div(nt, stepsPerCaseBlock);
		int caseblk = res.rem;
		res = div(caseblk, stepsPerCase);
		int caseID = res.quot;
		if (caseID > 0) {
			int xxx = 0;
		}
		unsignedToBinary(caseID, caseValue);
	}

};

bool Environment::booleanFunction(vector<bool> x) {
	return (x[0] ^ x[1]);  // XOR
}
