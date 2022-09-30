#pragma once
#include "pch.h"


struct BaseGroup {

	// Collection of States
	state_t mStates;				// collection of states					
	unsigned mSizeStates;			// total size of states
	unsigned mNumStates;			// number of states											
	string mName;
	unsigned mID;

	// Single State
	vector<value_t> mState;			// a single state
	unsigned mSizeState;			// size of a single state
	vector<string> mLabel;

	// ctor when the number of objects (NumStates) is known
	BaseGroup(string Name, unsigned ID, unsigned NumStates) :
		mName(Name),
		mID(ID),
		mNumStates(NumStates)
	{
		mLabel = { "ID" };
		mSizeState = mLabel.size();
		mSizeStates = mSizeState * mNumStates;
		mState.resize(mSizeState);
		mStates.resize(mSizeStates);
	}

	// ctor when the number of objects (NumStates) is not known
	BaseGroup(string Name, unsigned ID) :
		mName(Name),
		mID(ID)
	{
		mLabel = { "ID" };
		mSizeState = mLabel.size();

	}

	// dtor
	~BaseGroup() {

	}

	virtual void InitState(unsigned idx, state_t &x) = 0;
	virtual void ResetState(unsigned idx, state_t &x, value_t* data) = 0;
	virtual void StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) = 0;

	void Init();
	void Reset(value_t* data);
	void Step(value_t* data, value_t t, value_t dt);

	void Print(value_t* data, ofstream& ofs, bool hdr);
	void Printf(value_t* data);
	void Pack(value_t* data, state_t x, unsigned idx);
	void UnPack(value_t* data, state_t &x, unsigned idx);
	//void Pack(value_t* data, state_t x, unsigned idx, unsigned sizeState);
	//void UnPack(value_t* data, state_t &x, unsigned idx, unsigned sizeState);
};

void BaseGroup::InitState(unsigned idx, state_t &x) {}
void BaseGroup::ResetState(unsigned idx, state_t &x, value_t* data) {}
void BaseGroup::StepState(unsigned idx, state_t &x, value_t* data, value_t t, value_t dt) {}

void BaseGroup::Pack(value_t* data, state_t x, unsigned indx) {
	unsigned offset = indx * mSizeState;
	for (unsigned idx = 0; idx < mSizeState; idx++) {
		data[offset + idx] = x[idx];
	}

	if (mName == "EnvOutput") {
		cout << "Pack...indx:  " << indx << " ID:  " << x[0] << " INP:  " << x[1] << " X:  " << x[2] << endl;
	}
}
//void BaseGroup::Pack(value_t* data, state_t x, unsigned indx, unsigned sizeState) {
//	unsigned offset = indx * sizeState;
//	for (unsigned idx = 0; idx < sizeState; idx++) {
//		data[offset + idx] = x[idx];
//	}
//
//	if (mName == "EnvOutput") {
//		cout << "Pack2...indx:  " << indx << " ID:  " << x[0] << " INP:  " << x[1] << " X:  " << x[2] << endl;
//	}
//}

void BaseGroup::UnPack(value_t* data, state_t &x, unsigned indx) {
	unsigned offset = indx * mSizeState;
	for (unsigned idx = 0; idx < mSizeState; idx++) {
		x[idx] = data[offset + idx];
	}
	if (mName == "EnvOutput") {
		cout << "UnPack...indx:  " << indx << " ID:  " << x[0] << " INP:  " << x[1] << " X:  " << x[2] << endl;
	}
}

//void BaseGroup::UnPack(value_t* data, state_t &x, unsigned indx, unsigned sizeState) {
//	unsigned offset = indx * sizeState;
//	for (unsigned idx = 0; idx < sizeState; idx++) {
//		x[idx] = data[offset + idx];
//	}
//	if (mName == "EnvOutput") {
//		cout << "UnPack2...indx:  " << indx << " ID:  " << x[0] << " INP:  " << x[1] << " X:  " << x[2] << endl;
//	}
//}

void BaseGroup::Init() {

	// define local state variables
	unsigned offset;
	vector<value_t> x(mSizeState);

	for (unsigned indx = 0; indx < mNumStates; indx++) {

		// unpack state
		offset = indx * mSizeState;

		// compute init value
		x[0] = (value_t)indx;
		InitState(indx, x);

		// pack state
		for (unsigned idx = 0; idx < mSizeState; idx++) {
			mStates[offset + idx] = x[idx];
		}
	}
}

void BaseGroup::Reset(value_t* data) {

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

void BaseGroup::Step(value_t* data, value_t t, value_t dt) {

	// define local state variables
	vector<value_t> x(mSizeState);

	for (unsigned indx = 0; indx < mNumStates; indx++) {

		// unpack state
		UnPack(data, x, indx);

		// compute step value
		StepState(indx, x, data, t, dt);

		// pack state
		Pack(data, x, indx);
	}
}


void BaseGroup::Print(value_t* data, ofstream& ofs, bool hdr) {

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

void BaseGroup::Printf(value_t* data) {

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
