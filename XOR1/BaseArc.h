#pragma once
#include "pch.h"


struct BaseArc {

	// Collection of States
	state_t mStates;				// collection of states					
	unsigned mSizeStates;			// total size of states
	unsigned mNumStates;			// number of states											
	string mName;
	unsigned mID;

	// State enum
	enum State_IDX { ID_idx, EXC_idx, PRED_idx, SUCC_idx };

	// Single State
	vector<value_t> mState;			// a single state
	unsigned mSizeState;			// size of a single state
	vector<string> mLabel;

	// Arc Generation
	unsigned mPredID, mSuccID;
	RandomArcModel mArcGenerationModel;
	value_t mProbExcitatory;
	unsigned mOutDegree;
	bool mDirected;
	bool mVolumetric;

	// ctor
	BaseArc(string Name, unsigned ID, unsigned predID, unsigned succID,
		value_t ProbExcitatory, unsigned OutDegree, bool Directed, bool Volumetric,
		RandomArcModel ArcGenerationModel) :
		mName(Name),
		mID(ID),
		mPredID(predID),
		mSuccID(succID),
		mProbExcitatory(ProbExcitatory),
		mOutDegree(OutDegree),
		mDirected(Directed),
		mVolumetric(Volumetric)
	{
		mLabel = { "ID" };
		mLabel.push_back("EXC");
		mLabel.push_back("PRED");
		mLabel.push_back("SUCC");
		mSizeState = mLabel.size();


		mArcGenerationModel = degreeModel;
		if (ArcGenerationModel == degreeModel) {
			mArcGenerationModel = degreeModel;
		}
		if (ArcGenerationModel == probModel) {
			mArcGenerationModel = probModel;
		}

	}

	// dtor
	~BaseArc() {

	}

	virtual void InitState(unsigned idx, state_t &x) = 0;
	virtual void ResetState(unsigned idx, state_t &x, value_t* data) = 0;
	virtual void StepState(unsigned idx, state_t &x, 
		value_t* data, BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt) = 0;

	void Init(BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc) {};
	void Reset(value_t* data);
	void Step(value_t* data, BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt);

	void Print(value_t* data, ofstream& ofs, bool hdr);
	void Printf(value_t* data);
	void Pack(value_t* data, state_t x, unsigned idx);
	void UnPack(value_t* data, state_t &x, unsigned idx);
};

void BaseArc::InitState(unsigned idx, state_t &x) {}
void BaseArc::ResetState(unsigned idx, state_t &x, value_t* data) {}
void BaseArc::StepState(unsigned idx, state_t &x,
	value_t* data, BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt) {}

void BaseArc::Pack(value_t* data, state_t x, unsigned indx) {
	unsigned offset = indx * mSizeState;
	for (unsigned idx = 0; idx < mSizeState; idx++) {
		data[offset + idx] = x[idx];
	}
}


void BaseArc::UnPack(value_t* data, state_t &x, unsigned indx) {
	unsigned offset = indx * mSizeState;
	for (unsigned idx = 0; idx < mSizeState; idx++) {
		x[idx] = data[offset + idx];
	}
}



//void BaseArc::Init(BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc) {
//	// ID, EXC, PRED, SUCC
//	unsigned RandomArc, RandomExc;
//	unsigned indx = 0, numRemain, samp;
//	int ctr, exc;
//	bool sameNodeGroup = (nodePred.mID == nodeSucc.mID);
//	vector<value_t> x(mSizeState);
//	vector<unsigned> sample(nodeSucc.mNumStates);
//
//	if (mArcGenerationModel == degreeModel) {
//		sample = nodeSucc.startSample;
//		for (unsigned pred = 0; pred < nodePred.mNumStates; pred++) {
//			sample = nodePred.startSample;
//			if (sameNodeGroup) sample[pred] = 0;  // no self loops
//			numRemain = nodeSucc.mNumStates;
//			for (unsigned s = 0; s < mOutDegree; s++) {
//				uniform_int_distribution<int> uniform_dist(0, numRemain - 1);
//				samp = uniform_dist(rng);
//				ctr = -1;
//				// look for (samp)th element of sample that is equal to 1
//				for (unsigned i = 0; i < nodeSucc.mNumStates; i++) {
//					if (sample[i] == 1) ctr++;
//					if (ctr == samp) {
//						sample[i] = 2;  // chosen
//						break;
//					}
//				}
//				numRemain = numRemain - 1;
//			}
//
//			for (unsigned succ = 0; succ < nodeSucc.mNumStates; succ++) {
//				if (sample[succ] == 2) {
//					RandomExc = (unifdistribution(rng) < mProbExcitatory);
//					exc = 2 * RandomExc - 1;
//					x[0] = (value_t)indx;
//					x[1] = (value_t)exc;
//					x[2] = (value_t)pred;
//					x[3] = (value_t)succ;
//					InitState(indx, x);  // returns with more states in x
//					mStates.insert(mStates.end(), x.begin(), x.end());
//					indx++;
//				}
//			}
//		}
//	}
//
//	mNumStates = indx;
//	mSizeStates = mSizeState * mNumStates;
//	mState.resize(mSizeState);
//	mStates.resize(mSizeStates);
//}

void BaseArc::Reset(value_t* data) {

	// define local state variables
	vector<value_t> x(mSizeState);

	for (unsigned indx = 0; indx < mNumStates; indx++) {

		// unpack state

		// compute reset value
		x[0] = (value_t)indx;
		ResetState(indx, x, data);

		// pack state
		Pack(data, x, indx);
	}
}

void BaseArc::Step(value_t* data, BaseNodeGroup &nodePred, BaseNodeGroup &nodeSucc, value_t t, value_t dt) {

	// define local state variables
	vector<value_t> x(mSizeState);

	for (unsigned indx = 0; indx < mNumStates; indx++) {

		// unpack state
		UnPack(data, x, indx);

		// compute step value
		StepState(indx, x, data, nodePred, nodeSucc, t, dt);

		// pack state
		Pack(data, x, indx);
	}
}


void BaseArc::Print(value_t* data, ofstream& ofs, bool hdr) {

	// define local state variables
	unsigned offset;

	if (hdr) {
		ofs << ",NAME,GrpID";
		for (unsigned idx = 0; idx < mNumStates; idx++) {
			for (unsigned jdx = 0; jdx < mSizeState; jdx++) {
				ofs << "," << mLabel[jdx] << idx;
			}
		}
	}
	else {
		ofs << "," << mName << "," << mID;
		for (unsigned idx = 0; idx < mNumStates; idx++) {
			offset = idx * mSizeState;
			for (unsigned jdx = 0; jdx < mSizeState; jdx++) {
				ofs << "," << data[offset + jdx];
			}
		}
	}
}

void BaseArc::Printf(value_t* data) {

	// define local state variables
	unsigned offset;
	value_t x;
	// headers
	string hdr = mName + " " + to_string(mID) + "\n";
	printf(hdr.c_str());
	string lbl;
	for (unsigned idx = 0; idx < mSizeState; idx++) {
		lbl = lbl + " " + mLabel[idx];
	}
	printf(lbl.c_str());
	// data
	for (unsigned idx = 0; idx < mNumStates; idx++) {
		offset = idx * mSizeState;
		for (unsigned jdx = 0; jdx < mSizeState; jdx++) {
			x = data[offset + jdx];
			printf(" %f", x);
		}
		printf("\n");
	}
}

