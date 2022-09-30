#pragma once
#include "pch.h"

struct Agent {

	enum State_IDX { ID_idx, I_idx, V_idx, IOUT_idx, SPIKE_idx, SPIKET_idx };

	value_t mVThresh = -50.0;  // mV
	value_t mVRest = -65.0;   // mV
	value_t mVPeak = 50.0;     // mV
	value_t mVReset = -80.0;   // mV
	value_t mRefractoryPeriod = 5.0;  // ms
	value_t mR;					// KOhm cm^2
	value_t mC = 1.0;			// pF / cm^2
	value_t mTimeConst = 10.0;  // ms
	value_t mConductanceScale;	// mS / cm^2

	Agent() {
		mR = mTimeConst / mC;		// kOhm/cm^2
		mConductanceScale = 1.0;  // mS / cm^2 (100 pS/synapse and synapse = 10E-9 cm^2)
	}
	void QIFInitState(state_t &x);
	void QIFResetState(state_t &x);
	void QIFStepState(state_t &x, value_t t, value_t dt);

};

void Agent::QIFStepState(state_t &x, value_t t, value_t dt) {
	// ID, INP, X, INPOUT, SPIKE, SPIKET

	// unpack
	value_t I = x[I_idx];
	value_t V = x[V_idx];
	value_t Iout = x[IOUT_idx];
	value_t spike = x[SPIKE_idx];
	value_t spikeTime = x[SPIKET_idx];
	cout << "QIFStep:  " << " t:  " << t << " I:  " << I << " V:  " << V << endl;

	// define local variables
	value_t dV, term1 = 0.0, term2 = 0.0;
	value_t spikeDelTime;

	// compute
	spike = 0.0;
	spikeDelTime = t - spikeTime;
	if (spikeDelTime > mRefractoryPeriod) {
		term1 = (V - mVRest) * (V - mVThresh) / (mVThresh - mVRest);  // mV
		term2 = mR * mConductanceScale * I;  // mV
		dV = dt / 2.0 / mTimeConst * (term1 - term2);
		V = V + dV;
		term1 = (V - mVRest) * (V - mVThresh) / (mVThresh - mVRest);  // mV
		term2 = mR * mConductanceScale * I;  // mV
		dV = dt / 2.0 / mTimeConst * (term1 - term2);
		V = V + dV;
		if (V > mVPeak) {
			V = mVReset;
			spike = 1.0;
			spikeTime = t;
		}
		if (V < mVReset) {
			V = mVReset;
		}
	}

	// pack state

	x[1] = 0.0;
	x[2] = V;
	x[3] = I;
	x[4] = spike;
	x[5] = spikeTime;
}

void Agent::QIFInitState(state_t &x) {
	// ID, INP, X, INPOUT, SPIKE, SPIKET
	// unpack
	value_t inputCurrent;
	value_t voltage;
	value_t spike;
	value_t spikeTime;

	// compute

	inputCurrent = 0.0;
	voltage = mVRest;
	spike = 0.0;
	spikeTime = -100.0;

	// pack
	x[1] = inputCurrent;
	x[2] = voltage;
	x[3] = inputCurrent;
	x[4] = spike;
	x[5] = spikeTime;
}
void Agent::QIFResetState(state_t &x) {
	// ID, INP, X, SPIKE, SPIKET
	// unpack
	value_t inputCurrent;
	value_t voltage;
	value_t spike;
	value_t spikeTime;

	// compute
	inputCurrent = 0.0;
	voltage = mVRest;
	spike = 0.0;
	spikeTime = -100.0;  // large time so exceeds refractory period

	// pack
	x[1] = inputCurrent;
	x[2] = voltage;
	x[3] = inputCurrent;
	x[4] = spike;
	x[5] = spikeTime;
}
