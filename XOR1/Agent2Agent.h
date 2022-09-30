#pragma once
#include "pch.h"
#include "BaseGroup.h"
#include "BaseArcGroup.h"
#include "BaseLearner.h"

struct Agent2Agent : public BaseArcGroup
{

	BaseLearner* pLearner;

	enum State_IDX { ID_idx, EXC_idx, PRED_idx, SUCC_idx, W_idx, NC_idx, ELPLUS_idx, ELMINUS_idx };

	// Difference of exponentials model for normalized conductance of EPSP
	value_t mTauM = 1.5;	// Longer Time Constant 1.5
	value_t mTauR = 0.09;	// Shorter Time Constant 0.09
	value_t mB;				// Coefficient in normalized conductance

	// Reversal Potentials
	value_t mEexc = 0.0;	// Reversal Potential for Excitatory Synapses
	value_t mEinh = -70.0;	// Reversal Potential for Inhibitory Synapses

	// ctor
	Agent2Agent(string Name, unsigned ID, unsigned predID, unsigned succID,
		value_t ProbExcitatory, unsigned OutDegree, bool Directed, bool Volumetric,
		RandomArcModel ArcGenerationModel, value_t Weight, Environment *Env, BaseLearner *Learner) :
		BaseArcGroup(Name, ID, predID, succID, ProbExcitatory, OutDegree, 
			Directed, Volumetric, ArcGenerationModel, Weight), pLearner(Learner)
	{
		mLabel.push_back("NC");  // normalized post-synaptic conductance
		mLabel.push_back("ELPLUS");  // forward eligibility
		mLabel.push_back("ELMINUS");  // backward eligibility
		mLabel.push_back("STDP");  // filtered STDP
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
void Agent2Agent::InitState(unsigned idx, state_t &x) {
	// ID, EXC, PRED, SUCC, W, NC

	// mTauM > mTauR
	value_t temp(pow(mTauR / mTauM, mTauR / (mTauM - mTauR)) -
		         pow(mTauR / mTauM, mTauM / (mTauM - mTauR)));
	mB = 1 / temp;

	value_t nc = 0.0;

	// pack state of arc
	x[5] = nc;
	x[6] = 0.0;
	x[7] = 0.0;
	x[8] = 0.0;
}
void Agent2Agent::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, EXC, PRED, SUCC, W, NC 

	value_t nc = 0.0;

	// pack state of arc
	x[5] = nc;
	x[6] = 0.0;
	x[7] = 0.0;
	x[8] = 0.0;
}
void Agent2Agent::StepState(unsigned idx, state_t &x, value_t* data, 
	BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt) {
	// ID, EXC, PRED, SUCC, W, NC
	// unpack state
	int exc = x[1];
	unsigned pred = x[2];
	unsigned succ = x[3];
	value_t w = x[4];
	value_t nc = x[5];  // normalized post-synaptic conductance

	// pred data
	unsigned sizeStatePred = nodePred.mSizeState;
	state_t xPred(sizeStatePred);
	nodePred.UnPack(nodePred.mStates.data(), xPred, pred);
	value_t VPred = xPred[2];  // pred node output voltage
	value_t SPIKEPred = xPred[4];  // Spike Indicator
	value_t TPred = xPred[5];  // time of last spike

	// succ data
	//BaseNodeGroup mSuccV = *mNodeGroups[mSuccGroupID];
	unsigned sizeStateSucc = nodeSucc.mSizeState;
	state_t xSucc(sizeStateSucc);
	sizeStateSucc = nodeSucc.mSizeState;
	nodeSucc.UnPack(nodeSucc.mStates.data(), xSucc, succ);
	value_t ISucc = xSucc[1];  // succ node input current
	value_t VSucc = xSucc[2];  // succ node membrane potential
	value_t SPIKESucc = xSucc[4];  // Spike Indicator
	value_t TSucc = xSucc[5];  // time of last spike

	// reward data
	

	// define local variables
	value_t dI;
	value_t Erev;
	state_t ex(2, 0.0);
	value_t dtSpike;

	// compute
	// compute current into successor node
	ex[0] = SPIKEPred;
	ex[1] = SPIKESucc;
	dtSpike = t - TPred;
	nc = 0.0;
	if(dtSpike < 10.0 && dtSpike >= 0.0) nc = mB * (exp(-dtSpike / mTauM) - exp(-dtSpike / mTauR) );
	//if (nc < 1.0E-6) nc = 0.0;
	Erev = mEinh;
	if (exc) Erev = mEexc;

	dI = w * nc * (VSucc - Erev);
 	ISucc = ISucc + dI;

	// compute weight update
	pLearner->Step(x, ex, t, dt);
	
	cout << "Ag2AgStep:  " << " t:  " << t << " I:  " << ISucc << " dI:  " << dI << endl;
	// pack state of successor node
	xSucc[1] = ISucc;
	nodeSucc.Pack(nodeSucc.mStates.data(), xSucc, succ);

	// pack state of arc
	x[5] = nc;
}



//void BaseArcGroup::Init() {
//	unsigned RandomArc, RandomExc;
//	unsigned idx = 0, numRemain, samp;
//	int ctr;
//	bool sameNodeGroup = (predV.mID == succV.mID);
//	vector<unsigned> sample(succV.mNumStates);
//
//
//	//if (arcGenerationModel == probModel && sameNodeGroup) {
//	//	for (unsigned pred = 0; pred < predV.NumNode; pred++) {
//	//		for (unsigned succ = 0; succ < succV.NumNode; succ++) {
//	//			if (pred != succ) {
//	//				RandomArc = (unifdistribution(rng) < ProbArc);
//	//				if (RandomArc > 0) {
//	//					RandomExc = (unifdistribution(rng) < ProbExcitatory);
//	//					ID.push_back(idx); // ID
//	//					EXC.push_back(2 * RandomExc - 1);  // Arc Type
//	//					PRED.push_back(pred);  // Pred Node
//	//					SUCC.push_back(succ);  // Succ Node
//	//					idx++;
//	//				}
//	//			}
//	//		}
//	//	}
//	//}
//	//if (arcGenerationModel == probModel && !sameNodeGroup) {
//	//	for (unsigned pred = 0; pred < predV.NumNode; pred++) {
//	//		for (unsigned succ = 0; succ < succV.NumNode; succ++) {
//	//			RandomArc = (unifdistribution(rng) < ProbArc);
//	//			if (RandomArc > 0) {
//	//				RandomExc = (unifdistribution(rng) < ProbExcitatory);
//	//				ID.push_back(idx); // ID
//	//				EXC.push_back(2 * RandomExc - 1);  // Arc Type
//	//				PRED.push_back(pred);  // Pred Node
//	//				SUCC.push_back(succ);  // Succ Node
//	//				idx++;
//	//			}
//	//		}
//	//	}
//	//}
//	if (arcGenerationModel == degreeModel && sameNodeGroup) {
//		sample = succV.startSample;
//		for (unsigned pred = 0; pred < predV.mNumStates; pred++) {
//			sample = succV.startSample;
//			sample[pred] = 0;  // no self loops
//			numRemain = succV.mNumStates;
//			for (unsigned s = 0; s < outDegree; s++) {
//				uniform_int_distribution<int> uniform_dist(0, numRemain - 1);
//				samp = uniform_dist(rng);
//				ctr = -1;
//				// look for (samp)th element of sample that is equal to 1
//				for (unsigned i = 0; i < succV.mNumStates; i++) {
//					if (sample[i] == 1) ctr++;
//					if (ctr == samp) {
//						sample[i] = 2;  // chosen
//						break;
//					}
//				}
//				numRemain = numRemain - 1;
//			}
//
//			for (unsigned succ = 0; succ < succV.mNumStates; succ++) {
//				if (sample[succ] == 2) {
//					RandomExc = (unifdistribution(rng) < ProbExcitatory);
//					ID.push_back(idx); // ID
//					EXC.push_back(2 * RandomExc - 1);  // Arc Type
//					PRED.push_back(pred);  // Pred Node
//					SUCC.push_back(succ);  // Succ Node
//					idx++;
//				}
//			}
//		}
//	}
//	if (arcGenerationModel == degreeModel && !sameNodeGroup) {
//		sample = succV.startSample;
//		for (unsigned pred = 0; pred < predV.mNumStates; pred++) {
//			sample = succV.startSample;
//			numRemain = succV.mNumStates;
//			for (unsigned s = 0; s < outDegree; s++) {
//				uniform_int_distribution<int> uniform_dist(0, numRemain - 1);
//				samp = uniform_dist(rng);
//				ctr = -1;
//				// look for (samp)th element of sample that is equal to 1
//				for (unsigned i = 0; i < succV.mNumStates; i++) {
//					if (sample[i] == 1) ctr++;
//					if (ctr == samp) {
//						sample[i] = 2;  // chosen
//						break;
//					}
//				}
//				numRemain = numRemain - 1;
//			}
//
//			for (unsigned succ = 0; succ < succV.mNumStates; succ++) {
//				if (sample[succ] == 2) {
//					RandomExc = (unifdistribution(rng) < ProbExcitatory);
//					ID.push_back(idx); // ID
//					EXC.push_back(2 * RandomExc - 1);  // Arc Type
//					PRED.push_back(pred);  // Pred Node
//					SUCC.push_back(succ);  // Succ Node
//					idx++;
//				}
//			}
//		}
//	}
//
//	NumArc = idx;
//	arcVector = makeArcVector();
//}

//__global__ void BaseArcStep_GPU(value_t* ArcData, value_t* NodeData, unsigned ArcLength, unsigned NodeLength,
//	value_t ForceAlpha) {
//
//	// Constants
//	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Unpack
//	unsigned idx = tid * ArcLength;
//	unsigned ID = (unsigned)ArcData[idx];
//	int ATTR = (int)ArcData[idx + 1];
//	unsigned PRED = (unsigned)ArcData[idx + 2];
//	unsigned SUCC = (unsigned)ArcData[idx + 3];
//	value_t ALPH = ForceAlpha;
//	value_t Xpred = NodeData[PRED * NodeLength + 1];
//	value_t Xsucc = NodeData[SUCC * NodeLength + 1];
//
//	// Dynamics
//	value_t DFXpred = ATTR * ALPH * (Xpred - Xsucc);
//	value_t DFXsucc = -ATTR * ALPH * (Xpred - Xsucc);
//
//	// Pack
//	atomicAdd(&(NodeData[PRED * NodeLength + 3]), DFXpred);
//	atomicAdd(&(NodeData[SUCC * NodeLength + 3]), DFXsucc);
//
//	//__syncthreads();
//}



