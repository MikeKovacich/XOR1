#pragma once
#include "pch.h"
#include "Environment.h"
#include "BaseNodeGroup.h"

struct EnvOutput : public BaseNodeGroup {
	state_t X; 

	Environment* Env;

	EnvOutput(string nameGroup_, unsigned groupID_, unsigned NumNode_, Environment *Env_) :
		BaseNodeGroup(nameGroup_, groupID_, NumNode_), Env(Env_) {
		X.resize(NumNode, 0.0);
	};

	state_t makeNodeVector();
	void Init();
	void Step_CPU(value_t* nodeGroupData, value_t t, value_t dt);

	void print(value_t* NodeData, ofstream& ofs, value_t t, bool hdr);
	void printData(value_t* NodeData);
};

void EnvOutput::Step_CPU(value_t* nodeGroupData, value_t t, value_t dt) {
	vector<bool> caseValue(2, 0);
	Env->Step_CPU(t, dt, caseValue);
	for (int tid = 0; tid < NumNode; tid++) {

		// Unpack
		unsigned idx = tid * NodeLength;
		unsigned IDval = (unsigned)nodeGroupData[idx];
		value_t Xval = nodeGroupData[idx + 1];

		// Delta State


		// the dynamical equation with boundary checking
		Xval = caseValue[tid];
		//cout << "EnvOutput Step CPU  t:  " << t << "  Name:  " << nameGroup 
		//	<< " ID:  " << IDval << " X:  " << Xval << endl;

		// Pack
		nodeGroupData[idx + 1] = Xval;
	}
}

state_t EnvOutput::makeNodeVector() {
	state_t NodeVector;
	state_t NodeArray;
	for (unsigned i = 0; i < NumNode; i++) {
		NodeArray = { (value_t)ID[i], X[i] };
		NodeVector.insert(NodeVector.end(), NodeArray.begin(), NodeArray.end());
	}
	NodeLength = NodeArray.size();
	return NodeVector;
}

void EnvOutput::Init() {

	for (unsigned i = 0; i < NumNode; i++) {
		ID[i] = i;
		X[i] = 0.0;
	}
	nodeVector = makeNodeVector();
}

void EnvOutput::print(value_t* NodeData, ofstream& ofs, value_t t, bool hdr)
{
	unsigned idx, IDval;
	value_t Xval;
	if (hdr) {
		ofs << "NAME,GrpID";
		for (unsigned i = 0; i < NumNode; i++) {
			ofs << "," << "ID" << i << "," << "X" << i;
		}
	}
	else {
		ofs << "," << nameGroup << "," << groupID;
		for (unsigned i = 0; i < NumNode; i++) {
			idx = i * NodeLength;
			IDval = (unsigned)NodeData[idx];
			Xval= NodeData[idx + 1];
			ofs << "," << IDval << "," << Xval;
		}
	}
}


void EnvOutput::printData(value_t* NodeData)
{
	value_t x;
	string hdr = "Node Data for " + nameGroup + "\n";
	printf(hdr.c_str());
	string lbl = "ID X \n";
	printf(lbl.c_str());
	for (unsigned i = 0; i < NumNode; i++) {
		for (unsigned j = 0; j < NodeLength; j++) {
			x = NodeData[i * NodeLength + j];
			printf(" %f", x);
		}
		printf("\n");
	}
}

