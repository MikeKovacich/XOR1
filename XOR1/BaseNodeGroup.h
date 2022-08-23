#pragma once
#include "pch.h"

struct BaseNodeGroup
{
	// State
	using Tuple = tuple<unsigned>;
	ustate_t ID;

	state_t nodeVector;
	unsigned NodeLength;

	// Parameters
	string nameGroup;
	unsigned groupID;
	unsigned NumNode;
	vector<unsigned> startSample;

	// ctor1
	BaseNodeGroup(unsigned groupID_, unsigned NumNode_) : groupID(groupID_), NumNode(NumNode_) {
		nameGroup = "Default";
		NodeLength = 0;
		ID.resize(NumNode, 0);
		startSample.resize(NumNode, 1);
	}

	// ctor2
	BaseNodeGroup(string nameGroup_, unsigned groupID_, unsigned NumNode_) :
		nameGroup(nameGroup_),
		groupID(groupID_),
		NumNode(NumNode_) {
		ID.resize(NumNode, 0);
		startSample.resize(NumNode, 1);
		NodeLength = 0;
	}

	virtual void Init();
	virtual void Step_CPU(value_t* nodeGroupData, value_t t, value_t dt);
	virtual void Reset();
	virtual state_t makeNodeVector();
	virtual state_t makeStateVector(unsigned idx);
	virtual void printStateVector(value_t* NodeData, ofstream& ofs, unsigned idx, bool hdr);
	virtual void print(value_t* NodeData, ofstream& ofs, bool hdr);
	virtual void printData(value_t* NodeData, string lbl);
};

state_t BaseNodeGroup::makeNodeVector() {
	state_t NodeVector;
	state_t NodeArray;
	for (unsigned i = 0; i < NumNode; i++) {
		//NodeArray = { (value_t)ID[i] };
		NodeArray = makeStateVector(i);
		NodeVector.insert(NodeVector.end(), NodeArray.begin(), NodeArray.end());
	}
	NodeLength = NodeArray.size();
	return NodeVector;
}

void BaseNodeGroup::printStateVector(value_t* NodeData, ofstream& ofs, unsigned idx, bool hdr) {
	unsigned indx, IDval;
	if (hdr) {
		ofs << "," << "ID" << idx;
	}
	else {
		indx = idx * NodeLength;
		IDval = (unsigned)NodeData[indx];
		ofs << "," << IDval;
	}
}

state_t BaseNodeGroup::makeStateVector(unsigned idx) {
	state_t StateVector = { (value_t)ID[idx] };
	return(StateVector);
}

void BaseNodeGroup::print(value_t* NodeData, ofstream& ofs, bool hdr)
{
	unsigned idx, IDval;
	if (hdr) {
		ofs << "NAME,GrpID";
		for (unsigned i = 0; i < NumNode; i++) {
			//ofs << "," << "ID" << i;
			printStateVector(NodeData, ofs, i, hdr);
		}
	}
	else {
		for (unsigned i = 0; i < NumNode; i++) {
			//idx = i * NodeLength;
			//IDval = (unsigned)NodeData[idx];
			//ofs << "," << IDval;
			printStateVector(NodeData, ofs, i, hdr);
		}
	}
}

void BaseNodeGroup::printData(value_t* NodeData, string lbl)
{
	value_t x;
	string hdr = "Node Data for " + nameGroup + "\n";
	printf(hdr.c_str());
	//string lbl = "ID \n";
	printf(lbl.c_str());
	for (unsigned i = 0; i < NumNode; i++) {
		for (unsigned j = 0; j < NodeLength; j++) {
			x = NodeData[i * NodeLength + j];
			printf(" %f", x);
		}
		printf("\n");
	}
}

void BaseNodeGroup::Init() {

	for (unsigned i = 0; i < NumNode; i++) {
		ID[i] = i;
	}
	nodeVector = makeNodeVector();
}
void BaseNodeGroup::Reset() {

	for (unsigned i = 0; i < NumNode; i++) {
		ID[i] = i;
	}
	nodeVector = makeNodeVector();
}


void BaseNodeGroup::Step_CPU(value_t* nodeGroupData, value_t t, value_t dt) {
	// Constants

	for (int tid = 0; tid < NumNode; tid++) {

		// Unpack
		unsigned idx = tid * NodeLength;
		unsigned IDval = (unsigned)nodeGroupData[idx];

		// Delta State


		// the dynamical equation with boundary checking
		//cout << "Node Step CPU  t:  " << t << "  Name:  " << nameGroup << " ID:  " << IDval << endl;

		// Pack

	}
}

//__global__ void BaseNodeStep_GPU(value_t* NodeData, unsigned NodeLength,
//	value_t Drift, value_t Diffusion, value_t BoxWidth, value_t dt, curandState *states) {
//
//	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Constants
//	value_t minPos = 0.0;
//	value_t maxPos = BoxWidth;
//	// Unpack
//	unsigned idx = tid * NodeLength;
//	unsigned ID = (unsigned)NodeData[idx];
//	value_t X = NodeData[idx + 1];
//	value_t FX = NodeData[idx + 2];
//	unsigned CTR = (unsigned)NodeData[idx + 3];
//	value_t M = NodeData[idx + 4];
//
//	// Random Number Generator
//	CTR++;
//	curand_init(tid, CTR, 0, &states[tid]);
//
//	// Delta State
//	value_t dX;
//
//	dX = Drift * dt + FX * dt;
//
//	dX += Diffusion * sqrt(dt) * curand_normal(&states[tid]);
//
//	dX = dX / M;
//
//	// the dynamical equation 
//	X = X + dX;
//	if (X < minPos) { X = minPos; }
//	if (X > maxPos) { X = maxPos; }
//
//	// Pack
//	NodeData[idx + 1] = X;
//	NodeData[idx + 2] = 0.0;
//	NodeData[idx + 3] = CTR;
//
//	//__syncthreads();
//
//}
