#pragma once

#include "pch.h"
#include "BaseGroup.h"
#include "BaseArcGroup.h"

struct AgOut2AgAct : public BaseArcGroup
{

	// ctor
	AgOut2AgAct(string Name, unsigned ID, unsigned predID, unsigned succID,
		value_t ProbExcitatory, unsigned OutDegree, bool Directed, bool Volumetric,
		RandomArcModel ArcGenerationModel, value_t Weight) :
		BaseArcGroup(Name, ID, predID, succID, ProbExcitatory, OutDegree, Directed, Volumetric, ArcGenerationModel, Weight)
	{

		mSizeState = mLabel.size();

		mArcGenerationModel = degreeModel;
		if (ArcGenerationModel == degreeModel) {
			mArcGenerationModel = degreeModel;
		}
		if (ArcGenerationModel == probModel) {
			mArcGenerationModel = probModel;
		}

	}
	virtual void InitState(unsigned idx, state_t &x);
	virtual void ResetState(unsigned idx, state_t &x, value_t* data);
	virtual void StepState(unsigned idx, state_t &x, value_t* data,
		BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt);
};

void AgOut2AgAct::InitState(unsigned idx, state_t &x) {
	// ID, EXC, PRED, SUCC, W, NC

	// unpack

	// local variables

	// compute

	// pack

}
void AgOut2AgAct::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, EXC, PRED, SUCC, W, NC 

	// unpack

	// local variables

	// compute

	// pack

}
void AgOut2AgAct::StepState(unsigned idx, state_t &x, value_t* data,
	BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt) {
	// ID, EXC, PRED, SUCC, W, NC
	// unpack state
	int exc = x[1];
	unsigned pred = x[2];
	unsigned succ = x[3];
	value_t w = x[4];

	// pred data
	unsigned sizeStatePred = nodePred.mSizeState;
	state_t xPred(sizeStatePred);
	nodePred.UnPack(nodePred.mStates.data(), xPred, pred);
	value_t lhd = xPred[6];  // pred node output voltage = likelihood

	// succ data
	// ID, INP, X, INPOUT, NUMACTION, ACTION1, ... , ACTIONN
	//BaseNodeGroup mSuccV = *mNodeGroups[mSuccGroupID];
	unsigned sizeStateSucc = nodeSucc.mSizeState;
	state_t xSucc(sizeStateSucc);
	sizeStateSucc = nodeSucc.mSizeState;
	nodeSucc.UnPack(nodeSucc.mStates.data(), xSucc, succ);
	value_t ISucc = xSucc[1];  // succ node input current

	// local variables

	// compute
	lhd = w * lhd;
	ISucc = ISucc + lhd;  // likelihood sum

	// pack

	// pack state of successor node
	xSucc[1] = ISucc;
	xSucc[5 + idx] = lhd;
	nodeSucc.Pack(nodeSucc.mStates.data(), xSucc, succ);

	// pack state of arc

}
