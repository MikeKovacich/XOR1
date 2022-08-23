#pragma once

#include "pch.h"

struct BaseArcGroup
{
	ustate_t ID;
	intstate_t EXC;
	ustate_t PRED;
	ustate_t SUCC;

	state_t arcVector;
	unsigned ArcLength;

	// Parameters
	string nameGroup;
	BaseNodeGroup  predV;
	unsigned predGroupID;
	BaseNodeGroup  succV;
	unsigned succGroupID;
	RandomArcModel arcGenerationModel;
	unsigned NumArc;
	value_t ProbExcitatory;
	unsigned outDegree;
	bool directed;
	bool volumetric;



	// Default ctor
	BaseArcGroup(BaseNodeGroup predV_, unsigned predGroupID_, BaseNodeGroup  succV_, unsigned succGroupID_) :
		predV(predV_), 
		predGroupID(predGroupID_),
		succV(succV_),
		succGroupID(succGroupID_) {
		arcGenerationModel = degreeModel;
		ProbExcitatory = 0.8;
		outDegree = 1;
		directed = true;
		volumetric = false;
		nameGroup = predV.nameGroup + "->" + succV.nameGroup;
	}
	BaseArcGroup(BaseNodeGroup  predV_, unsigned predGroupID_, BaseNodeGroup  succV_, unsigned succGroupID_,
		value_t ProbExcitatory_, unsigned outDegree_, bool directed_, bool volumetric_,
		RandomArcModel arcGenerationModel_) :
		predV(predV_),
		predGroupID(predGroupID_),
		succV(succV_),
		succGroupID(succGroupID_),
		ProbExcitatory(ProbExcitatory_),
		outDegree(outDegree_),
		directed(directed_),
		volumetric(volumetric_)
	{
		arcGenerationModel = degreeModel;
		if (arcGenerationModel_ == degreeModel) {
			arcGenerationModel = degreeModel;
		}
		if (arcGenerationModel_ == probModel) {
			arcGenerationModel = probModel;
		}
		nameGroup = predV.nameGroup + "->" + succV.nameGroup;
	}

	virtual void Init();
	virtual void Step_CPU(value_t* arcGroupsData, value_t* predNodeGroupsData, value_t* succNodeGroupsData, value_t t, value_t dt);
	virtual void Reset();
	virtual state_t  makeArcVector();
	virtual void print(value_t* ArcData, ofstream ofs, value_t t, bool hdr);
	virtual void printData(value_t* ArcData);
};

state_t BaseArcGroup::makeArcVector() {
	state_t ArcVector;
	state_t ArcArray;
	for (unsigned i = 0; i < NumArc; i++) {
		ArcArray = { (value_t)ID[i], (value_t)EXC[i], (value_t)PRED[i], (value_t)SUCC[i] };
		ArcVector.insert(ArcVector.end(), ArcArray.begin(), ArcArray.end());
	}
	ArcLength = ArcArray.size();
	return ArcVector;
}

void BaseArcGroup::print(value_t* ArcData, ofstream ofs, value_t t, bool hdr)
{
	unsigned idx, IDval, EXCval, PREDval, SUCCval;
	if (hdr) {
		ofs << "NAME";
		for (unsigned i = 0; i < NumArc; i++) {
			ofs << "," << "ID" << i;
		}
	}
	else {
		ofs << "," << nameGroup;
		for (unsigned i = 0; i < NumArc; i++) {
			idx = i * ArcLength;
			IDval = (unsigned)ArcData[idx];
			ofs << t << " " << IDval;
		}
	}
}

void BaseArcGroup::printData(value_t* ArcData)
{
	value_t x;
	string hdr = "Arc Data for " + nameGroup + "\n";
	printf(hdr.c_str());
	string lbl = "ID EXC PRED SUCC \n";
	printf(lbl.c_str());
	for (unsigned i = 0; i < NumArc; i++) {
		for (unsigned j = 0; j < ArcLength; j++) {
			x = ArcData[i * ArcLength + j];
			printf(" %f", x);
		}
		printf("\n");
	}
}
void BaseArcGroup::Init() {
	unsigned RandomArc, RandomExc;
	unsigned idx = 0, numRemain, samp;
	int ctr;
	bool sameNodeGroup = (predV.groupID == succV.groupID);
	vector<unsigned> sample(succV.NumNode);


	//if (arcGenerationModel == probModel && sameNodeGroup) {
	//	for (unsigned pred = 0; pred < predV.NumNode; pred++) {
	//		for (unsigned succ = 0; succ < succV.NumNode; succ++) {
	//			if (pred != succ) {
	//				RandomArc = (unifdistribution(rng) < ProbArc);
	//				if (RandomArc > 0) {
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
	//}
	//if (arcGenerationModel == probModel && !sameNodeGroup) {
	//	for (unsigned pred = 0; pred < predV.NumNode; pred++) {
	//		for (unsigned succ = 0; succ < succV.NumNode; succ++) {
	//			RandomArc = (unifdistribution(rng) < ProbArc);
	//			if (RandomArc > 0) {
	//				RandomExc = (unifdistribution(rng) < ProbExcitatory);
	//				ID.push_back(idx); // ID
	//				EXC.push_back(2 * RandomExc - 1);  // Arc Type
	//				PRED.push_back(pred);  // Pred Node
	//				SUCC.push_back(succ);  // Succ Node
	//				idx++;
	//			}
	//		}
	//	}
	//}
	if (arcGenerationModel == degreeModel && sameNodeGroup) {
		sample = succV.startSample;
		for (unsigned pred = 0; pred < predV.NumNode; pred++) {
			sample = succV.startSample;
			sample[pred] = 0;  // no self loops
			numRemain = succV.NumNode;
			for (unsigned s = 0; s < outDegree; s++) {
				uniform_int_distribution<int> uniform_dist(0, numRemain - 1);
				samp = uniform_dist(rng);
				ctr = -1;
				// look for (samp)th element of sample that is equal to 1
				for (unsigned i = 0; i < succV.NumNode; i++) {
					if (sample[i] == 1) ctr++;
					if (ctr == samp) {
						sample[i] = 2;  // chosen
						break;
					}
				}
				numRemain = numRemain - 1;
			}

			for (unsigned succ = 0; succ < succV.NumNode; succ++) {
				if (sample[succ] == 2) {
					RandomExc = (unifdistribution(rng) < ProbExcitatory);
					ID.push_back(idx); // ID
					EXC.push_back(2 * RandomExc - 1);  // Arc Type
					PRED.push_back(pred);  // Pred Node
					SUCC.push_back(succ);  // Succ Node
					idx++;
				}
			}
		}
	}
	if (arcGenerationModel == degreeModel && !sameNodeGroup) {
		sample = succV.startSample;
		for (unsigned pred = 0; pred < predV.NumNode; pred++) {
			sample = succV.startSample;
			numRemain = succV.NumNode;
			for (unsigned s = 0; s < outDegree; s++) {
				uniform_int_distribution<int> uniform_dist(0, numRemain - 1);
				samp = uniform_dist(rng);
				ctr = -1;
				// look for (samp)th element of sample that is equal to 1
				for (unsigned i = 0; i < succV.NumNode; i++) {
					if (sample[i] == 1) ctr++;
					if (ctr == samp) {
						sample[i] = 2;  // chosen
						break;
					}
				}
				numRemain = numRemain - 1;
			}

			for (unsigned succ = 0; succ < succV.NumNode; succ++) {
				if (sample[succ] == 2) {
					RandomExc = (unifdistribution(rng) < ProbExcitatory);
					ID.push_back(idx); // ID
					EXC.push_back(2 * RandomExc - 1);  // Arc Type
					PRED.push_back(pred);  // Pred Node
					SUCC.push_back(succ);  // Succ Node
					idx++;
				}
			}
		}
	}

	NumArc = idx;
	arcVector = makeArcVector();
}

void BaseArcGroup::Reset() {


}

void BaseArcGroup::Step_CPU(value_t* arcGroupsData, value_t* predNodeGroupsData, value_t* succNodeGroupsData, value_t t, value_t dt) {
	// Constants
	unsigned predNodeLength = predV.NodeLength;
	unsigned succNodeLength = succV.NodeLength;
	//printData(arcGroupsData);
	for (unsigned tid = 0; tid < NumArc; tid++) {

		// Unpack
		unsigned idx = tid * ArcLength;
		unsigned IDval = (unsigned)arcGroupsData[idx];
		int ATTRval = (int)arcGroupsData[idx + 1];
		unsigned PREDval = (unsigned)arcGroupsData[idx + 2];
		unsigned SUCCval = (unsigned)arcGroupsData[idx + 3];

		// Dynamics
		//cout << "Arc Step CPU  t:  " << t << " Arc ID:  " << IDval << " Pred Grp Name:  " << predV.nameGroup << " Node ID:  " << PREDval
		//	<< "  Succ Grp Name:  " << succV.nameGroup << " Node ID:  " << SUCCval << endl;

		// Pack

	}
}

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

