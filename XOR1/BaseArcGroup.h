#pragma once
#include "pch.h"
#include "BaseArc.h"
#include "BaseNodeGroup.h"
#include "System.h"

struct BaseArcGroup : public BaseArc
{
	enum State_IDX { ID_idx, EXC_idx, PRED_idx, SUCC_idx, W_idx };

	value_t mWeight;


	// ctor
	BaseArcGroup(string Name, unsigned ID, unsigned predID, unsigned succID,
		value_t ProbExcitatory, unsigned OutDegree, bool Directed, bool Volumetric,
		RandomArcModel ArcGenerationModel, value_t Weight) :
		BaseArc(Name, ID, predID, succID, ProbExcitatory, OutDegree, Directed, Volumetric, ArcGenerationModel),
		mWeight(Weight)
	{
		mLabel.push_back("W");
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
	virtual void StepState(unsigned idx, state_t &x, 
		value_t* data, BaseNodeGroup *nodePred, BaseNodeGroup *nodeSucc, value_t t, value_t dt);

	virtual void Init(BaseNodeGroup *nodePred, BaseNodeGroup *nodeSucc);
	vector<vector<unsigned>> GenerateArcsDistributed(unsigned numRow, unsigned numCol,
		unsigned outDegree, bool sameGroup);
	vector<vector<unsigned>> GenerateArcsDistributedTry(unsigned numRow, unsigned numCol,
		unsigned outDegree, bool sameGroup, unsigned &rowFail, unsigned &colFail);
	vector<vector<unsigned>> GenerateArcsRandom(unsigned numRow, unsigned numCol,
		unsigned outDegree, bool sameGroup);
	void PrintArcs(unsigned numRow, unsigned numCol, vector<vector<unsigned>> arc);



};
void BaseArcGroup::InitState(unsigned idx, state_t &x) {
	// ID, EXC, PRED, SUCC, W 

}
void BaseArcGroup::ResetState(unsigned idx, state_t &x, value_t* data) {
	// ID, EXC, PRED, SUCC, W

}
void BaseArcGroup::StepState(unsigned indx, state_t &x, 
	value_t* data, BaseNodeGroup *nodePred, BaseNodeGroup *nodeSucc, value_t t, value_t dt) {
	// ID, EXC, PRED, SUCC, W
	// unpack state
	int exc = x[1];
	unsigned pred = x[2];
	unsigned succ = x[3];
	value_t w = x[4];


	// pred data

	unsigned sizeStatePred = nodePred->mSizeState;
	state_t xPred(sizeStatePred);
	nodePred->UnPack(nodePred->mStates.data(), xPred, pred);
	value_t VPred = xPred[2];  // pred node output voltage

	// succ data
	//BaseNodeGroup mSuccV = *mNodeGroups[mSuccGroupID];
	unsigned sizeStateSucc = nodeSucc->mSizeState;
	state_t xSucc(sizeStateSucc);
	sizeStateSucc = nodeSucc->mSizeState;
	nodeSucc->UnPack(nodeSucc->mStates.data(), xSucc, succ);
	value_t ISucc = xSucc[1];  // succ node input current

	// define local variables
	value_t dI;
	if (t >= 200.0) {
		int xx = 0;
	}
	// compute
	dI = -w * VPred;
	ISucc = ISucc + dI;

	// pack state of successor node
	// note:  no update of arc itself
	xSucc[1] = ISucc;

	nodeSucc->Pack(nodeSucc->mStates.data(), xSucc, succ);  
}

void BaseArcGroup::Init(BaseNodeGroup *nodePred, BaseNodeGroup *nodeSucc) {

	bool sameGroup = (nodePred->mID == nodeSucc->mID);
	unsigned numRow = nodePred->mNumStates;
	unsigned numCol = nodeSucc->mNumStates;
	unsigned outDegree = mOutDegree;
	bool distributedArcs = true;
	vector<vector<unsigned>> arc;

	if (distributedArcs) {
		arc = GenerateArcsDistributed(numRow, numCol, outDegree, sameGroup);
	}
	else {
		arc = GenerateArcsRandom(numRow, numCol, outDegree, sameGroup);
	}

	// set arcs
	vector<value_t> x(mSizeState);
	value_t exc;
	unsigned arcID = 0;
	for (unsigned row = 0; row < numRow; row++) {
		for (unsigned col = 0; col < numCol; col++) {
			if (arc[row][col] == 1) {
				exc = 2 * (unifdistribution(rng) < mProbExcitatory) - 1;
				x[0] = (value_t)arcID;
				x[1] = (value_t)exc;
				x[2] = (value_t)row;
				x[3] = (value_t)col;
				x[4] = mWeight;
				InitState(arcID, x);  // returns with more states in x
				mStates.insert(mStates.end(), x.begin(), x.end());
				arcID++;
			}
		}
	}

	mNumStates = arcID;
	mSizeStates = mSizeState * mNumStates;
	mState.resize(mSizeState);
	mStates.resize(mSizeStates);
}

vector<vector<unsigned>> BaseArcGroup::GenerateArcsRandom(unsigned numRow, unsigned numCol,
	unsigned outDegree, bool sameGroup) {

	// random number generator
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine rng(seed1);

	// local variables
	unsigned numArc = numRow * numCol;
	unsigned randomCol;
	int ctr;
	bool avail;

	// arcs
	vector<vector<unsigned>> arc(numRow);
	for (int i = 0; i < numRow; i++)
		arc[i].resize(numCol, 0);

	// capacities
	vector<unsigned> capacity(numRow, outDegree);

	// validation
	if (sameGroup) numCol = numRow;
	if (sameGroup) {
		if (outDegree > numCol - 1) {
			cout << "outDegree is too large." << endl;
			return(arc);
		}
	}
	else {
		if (outDegree > numCol) {
			cout << "outDegree is too large" << endl;
			return(arc);
		}
	}
	unsigned ranCol;
	bool updated;
	uniform_int_distribution<int> uniform_dist(0, numCol - 1);
	for (unsigned row = 0; row < numRow; row++) {
		for (unsigned d = 0; d < outDegree; d++) {
			do {
				updated = false;
				ranCol = uniform_dist(rng);
				if (arc[row][ranCol] == 0) {
					if (sameGroup) {
						if (row != ranCol) {
							arc[row][ranCol] = 1;
							updated = true;
						}
					}
					else {
						arc[row][ranCol] = 1;
						updated = true;
					}
				};
			} while (!updated);
		}
		return(arc);
	}
}

vector<vector<unsigned>> BaseArcGroup::GenerateArcsDistributed(unsigned numRow, unsigned numCol,
	unsigned outDegree, bool sameGroup) {
	vector<vector<unsigned>> arc;
	vector<vector<unsigned>> bestArc;
	unsigned lowestRowFail, lowestColFail, lowestFail;
	unsigned rowFail, colFail;
	lowestFail = 1000;
	unsigned maxTry = 20;
	for (unsigned i = 0; i < maxTry; i++) {
		arc = GenerateArcsDistributedTry(numRow, numCol, outDegree, sameGroup, rowFail, colFail);
		if (rowFail + colFail < lowestFail) {
			lowestRowFail = rowFail;
			lowestColFail = colFail;
			bestArc = arc;
		}
		if (rowFail == 0 && colFail == 0) {
			//cout << "try:  " << i + 1 << endl;
			break;
		}
	}
	arc = bestArc;
	return(arc);
}


vector<vector<unsigned>> BaseArcGroup::GenerateArcsDistributedTry(unsigned numRow, unsigned numCol,
	unsigned outDegree, bool sameGroup, unsigned &rowFail, unsigned &colFail) {
	// random number generator
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine rng(seed1);

	// local variables
	unsigned numArc = numRow * numCol;

	// arcs
	vector<vector<unsigned>> arc(numRow);
	for (int i = 0; i < numRow; i++)
		arc[i].resize(numCol, 0);

	// validation
	if (sameGroup) numCol = numRow;
	if (sameGroup) {
		if (outDegree > numCol - 1) {
			cout << "outDegree is too large." << endl;
			return(arc);
		}
	}
	else {
		if (outDegree > numCol) {
			cout << "outDegree is too large" << endl;
			return(arc);
		}
	}


	// capacities
	// Capacity of each row
	vector<unsigned> capacityRow(numRow, outDegree);
	vector<unsigned> capacityRowStart(capacityRow);
	vector<unsigned> capacityRowStart2(capacityRow);

	// Capacity of each col
	div_t divresult = div((int)numRow * outDegree, (int)numCol);
	unsigned inDegreeQ = divresult.quot;
	vector<unsigned> capacityCol(numCol, inDegreeQ);
	unsigned numLeft = numRow * outDegree - numCol * inDegreeQ;
	for (unsigned col = 0; col < numLeft; col++) {
		capacityCol[col]++;
	}
	vector<unsigned> capacityColStart(capacityCol);
	vector<unsigned> capacityColStart2(capacityCol);

	// Capacity of each (row, col) pair
	vector<vector<unsigned>> capacity(numRow);
	for (int i = 0; i < numRow; i++)
		capacity[i].resize(numCol, 0);
	for (unsigned row = 0; row < numRow; row++) {
		for (unsigned col = 0; col < numCol; col++) {
			capacity[row][col] = capacityRow[row] * capacityCol[col] + 0.01 * normaldistribution(rng);
			if ((row == col) && sameGroup) capacity[row][col] = 0;
			//if (row == (numCol - 1 - col) && !sameGroup) capacity[row][col] = 0;  //
		}
	}

	// attempt a solution
	bool addArc;
	int maxCapacity, maxColCapacity;
	unsigned maxRowIndx, maxColIndx;
	for (unsigned a = 0; a < numArc; a++) {
		maxCapacity = 0;
		for (unsigned row = 0; row < numRow; row++) {
			for (unsigned col = 0; col < numCol; col++) {
				if (capacity[row][col] >= maxCapacity) {
					maxCapacity = capacity[row][col];
					maxRowIndx = row;
					maxColIndx = col;
				}
			}
		}
		addArc = (capacityRow[maxRowIndx] > 0) && (capacityCol[maxColIndx] > 0);

		if (addArc) {
			arc[maxRowIndx][maxColIndx] = 1;
			capacity[maxRowIndx][maxColIndx] = 0;
			capacityRow[maxRowIndx]--;
			capacityCol[maxColIndx]--;
			for (unsigned row = 0; row < numRow; row++) {
				for (unsigned col = 0; col < numCol; col++) {
					capacity[row][col] = capacityRow[row] * capacityCol[col] * (1 - arc[row][col]);
					if ((row == col) && sameGroup) capacity[row][col] = 0;
					//if (row == (numCol - 1 - col) && !sameGroup) capacity[row][col] = 0;
				}
			}
		}
	}

	// test for validity
	unsigned sumVal;
	rowFail = 0;
	for (unsigned row = 0; row < numRow; row++) {
		sumVal = 0;
		for (unsigned col = 0; col < numCol; col++) {
			sumVal = sumVal + arc[row][col];
		}
		if (sumVal != capacityRowStart[row]) rowFail++;
	}
	colFail = 0;
	for (unsigned col = 0; col < numCol; col++) {
		sumVal = 0;
		for (unsigned row = 0; row < numRow; row++) {
			sumVal = sumVal + arc[row][col];
		}
		if (sumVal != capacityColStart[col]) colFail++;
	}
	cout << "Num Row Fails:  " << rowFail << " Num Col Fails:  " << colFail << endl;
	return(arc);
}


void BaseArcGroup::PrintArcs(unsigned numRow, unsigned numCol, vector<vector<unsigned>> arc) {
	cout << " ";
	for (unsigned col = 0; col < numCol; col++) {
		cout << "    " << col;
	}
	cout << endl;
	for (unsigned row = 0; row < numRow; row++) {
		cout << row;
		for (unsigned col = 0; col < numCol; col++) {
			cout << "    " << arc[row][col];
		}
		cout << endl;
	}
}
//void BaseArcGroup::Reset(value_t* data) {
//
//	// define local state variables
//	vector<value_t> xo(mSizeState);
//
//	for (unsigned idx = 0; idx < mNumStates; idx++) {
//
//		// unpack state
//
//		// compute reset value
//
//
//		// pack state
//
//	}
//}
//
//void BaseArcGroup::Step(value_t* data, value_t t, value_t dt) {
//
//	// define local state variables
//	vector<value_t> xi(mSizeState);
//	vector<value_t> xo(mSizeState);
//
//	for (unsigned idx = 0; idx < mNumStates; idx++) {
//
//		// unpack state
//		UnPack(data, xi, idx);
//		value_t ID = xi[0];
//		value_t ATT = xi[1];
//		value_t PRED = xi[2];
//		value_t SUCC = xi[3];
//
//
//		// compute step value
//
//		// pack state
//
//	}
//}
//
//





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

