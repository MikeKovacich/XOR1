#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"
#include "BaseArcGroup.h"
#include "Environment.h"
#include "Evaluation.h"
#include "AgentInput.h"
#include "AgentLayer1.h"
#include "AgentLayer2.h"
#include "AgentOutput.h"
#include "AgentAction.h"
#include "EnvHeat.h"
#include "EnvWork.h"
#include "EnvOutput.h"
#include "EnvReward.h"
#include "Agent2Agent.h"
#include "AgOut2AgAct.h"
#include "Learner1.h"


struct System{

	string systemJSONFile;
	stringstream buffer;
	string systemName;
	Environment* pEnv;
	Evaluation* pEval;
	vector<unique_ptr<BaseNodeGroup>> nodeGroups;
	vector<unique_ptr<BaseArcGroup>> arcGroups;
	map<string, unsigned> nodeGroupID;

	// ctor
	System(string systemJSONFile_) : systemJSONFile(systemJSONFile_) {};
	System() {}

	// dtor
	~System() {
	}

	// copy constructor
	System(const System &Sys);

	// construction
	void Build();
	void readSystemJSON();
	void buildEnvironmentFromJSON(unsigned numStepsPerCase);
	void buildEvaluationFromJSON();
	void buildNodesFromJSON();
	void buildArcsFromJSON();
	void PrintGraph(string GraphFileSpec);

	// dynamics
	void Init();
	void Reset();
	void Step(value_t t, value_t dt);
	void Print(ofstream &ofs, bool printHeader);

};

System::System(const System &Sys) {
	systemJSONFile = Sys.systemJSONFile;
}

void System::readSystemJSON() {

	// Get Configuration JSON file
	ifstream fJson(systemJSONFile);
	buffer << fJson.rdbuf();
	auto systemJSON = json::parse(buffer.str());
	systemJSON.at("System").at("Name").get_to(systemName);
}

void System::Build() {
	readSystemJSON();
	buildEvaluationFromJSON();
	buildEnvironmentFromJSON(pEval->numStepsPerCase);
	buildNodesFromJSON();
	buildArcsFromJSON();
}

void System::PrintGraph(string GraphFileSpec) {
	// print graph in GDF format
	// open file
	ofstream ofs(GraphFileSpec, ofstream::out);
	int graphFile_OK = ofs.is_open();

	// local variables
	value_t* nodeData, *nodeData1, *nodeData2;
	unsigned offset, offsetPred, offsetSucc, sizeState, sizeStatePred, sizeStateSucc, pred, succ;
	string nameGroup, nameGroupPred, nameGroupSucc;
	string node, nodePred, nodeSucc;
	unsigned ID, predGrpID, succGrpID, predID, succID;
	unsigned excitatory;
	string dirString, excString;


	// output graph

	ofs << "nodedef>name VARCHAR\n";
	// Loop over nodeGroups and then over each node in each group
	for (unsigned idx = 0; idx < nodeGroups.size(); idx++) {
		nameGroup = nodeGroups[idx]->mName;
		sizeState = nodeGroups[idx]->mSizeState;
		for (unsigned jdx = 0; jdx < nodeGroups[idx]->mNumStates; jdx++) {
			offset = jdx * sizeState;
			ID = nodeGroups[idx]->mStates[offset];
			node = nameGroup + "." + to_string(ID);
			ofs << node << '\n';
		}
	}
	ofs << "edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN,excitatory BOOLEAN" << '\n';
	for (unsigned idx = 0; idx < arcGroups.size(); idx++) {
		// get group data for PRED and SUCC
		sizeState = arcGroups[idx]->mSizeState;
		predGrpID = arcGroups[idx]->mPredID;
		succGrpID = arcGroups[idx]->mSuccID;
		nameGroupPred = nodeGroups[predGrpID]->mName;
		sizeStatePred = nodeGroups[predGrpID]->mSizeState;
		nameGroupSucc = nodeGroups[succGrpID]->mName;
		sizeStateSucc = nodeGroups[succGrpID]->mSizeState;
		// loop over arcs in arcGroup
		for (unsigned jdx = 0; jdx < arcGroups[idx]->mNumStates; jdx++) {
			offset = jdx * sizeState;
			// ID of pred node (prefixed by name of nodeGroup)
			pred = arcGroups[idx]->mStates[offset + 2];
			offsetPred = pred * sizeStatePred;
			predID = nodeGroups[predGrpID]->mStates[offsetPred];
			nodePred = nameGroupPred + '.' + to_string(predID);
			// ID of succ node (prefixed by name of nodeGroup)
			succ = arcGroups[idx]->mStates[offset + 3];
			offsetSucc = succ * sizeStateSucc;
			succID = nodeGroups[succGrpID]->mStates[offsetSucc];
			nodeSucc = nameGroupSucc + '.' + to_string(succID);
			// Directed or Not
			dirString = "true";
			// Excitatory or Not
			excitatory = (unsigned) arcGroups[idx]->mStates[offset + 1];
			excString = "false";
			if (excitatory == 1) excString = "true";
			ofs << nodePred << "," << nodeSucc << "," << dirString << "," << excString << '\n';
		}
	}
	// close file
	ofs.close();
}

void System::buildEvaluationFromJSON() {
	auto systemJSON = json::parse(buffer.str());
	auto EvaluationJSON = systemJSON.at("System").at("Evaluation");

	string Label = EvaluationJSON["Label"].get<string>();
	unsigned numStepsTraining = EvaluationJSON.at("Training").at("Number of Steps").get<int>();
	unsigned numTrialsTraining = EvaluationJSON.at("Training").at("Number of Trials").get<int>();
	unsigned numStepsPerCase = EvaluationJSON.at("Training").at("Number of Steps per Case").get<int>();
	unsigned numStepsTesting = EvaluationJSON.at("Testing").at("Number of Steps").get<int>();
	unsigned numTrialsTesting = EvaluationJSON.at("Testing").at("Number of Trials").get<int>();
	value_t dt = EvaluationJSON.at("Training").at("Step Size").get<value_t>();

	pEval = new Evaluation(Label, numStepsTraining, numTrialsTraining,
		numStepsTesting, numTrialsTesting, numStepsPerCase, dt);
}

void System::buildEnvironmentFromJSON(unsigned numStepsPerCase) {
	auto systemJSON = json::parse(buffer.str());
	auto EnvironmentJSON = systemJSON.at("System").at("Environment");

	string Label = EnvironmentJSON.at("Label").get<string>();
	unsigned numCases = EnvironmentJSON.at("Number of Cases").get<int>();
	string randomCasesString = EnvironmentJSON.at("Random Cases per Block").get<string>();
	bool randomCases;
	if (randomCasesString == "no") randomCases = false; else randomCases = true;
	pEnv = new Environment(Label, numStepsPerCase, numCases, randomCases);
}


void System::buildNodesFromJSON() {
	auto systemJSON = json::parse(buffer.str());
	auto NodesJSON = systemJSON.at("System").at("Nodes");
	
	string nameGroup;
	unsigned numNode;
	unsigned numAction = 0;
	unsigned groupID;
	param_t Params;
	for (int inode = 0; inode < NodesJSON.size(); inode++) {
		// unpack JSON
		nameGroup = NodesJSON[inode]["Name"].get<string>();
		numNode = NodesJSON[inode]["Number of Nodes"].get<int>();
		groupID = inode;
		nodeGroupID[nameGroup] = groupID;

		// create nodeGroup structures
		cout << "Building Node with Name:  " << nameGroup << endl;

		switch (groupID) {
		case 0:
			//AgentInput
		{
			AgentInput *pNodeGroup = new AgentInput(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		case 1:
			//AgentLayer1
		{
			AgentLayer1 *pNodeGroup = new AgentLayer1(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 2:
			//AgentLayer2
		{
			AgentLayer2 *pNodeGroup = new AgentLayer2(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 3:
			//AgentOutput
		{
			AgentOutput *pNodeGroup = new AgentOutput(nameGroup, groupID, numNode);
			numAction = numNode;
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 4:
			//AgentAction
		{
			AgentAction *pNodeGroup = new AgentAction(nameGroup, groupID, numNode, numAction);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 5:
			// EnvHeat
		{
			EnvHeat *pNodeGroup = new EnvHeat(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 6:
			// EnvWork
		{
			EnvWork *pNodeGroup = new EnvWork(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		break;
		case 7:
			// EnvOutput
		{
			EnvOutput *pNodeGroup = new EnvOutput(nameGroup, groupID, numNode, pEnv);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		case 8:
			// EnvReward
		{
			Params["gammaR" ] = 0.99;
			EnvReward *pNodeGroup = new EnvReward(nameGroup, groupID, numNode, pEnv, Params);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
			break;
		}
		default:
		{
			BaseNodeGroup *pNodeGroup = new BaseNodeGroup(nameGroup, groupID, numNode);
			pNodeGroup->Init();
			nodeGroups.emplace_back(pNodeGroup);
		}
		}	
	}
}

void System::buildArcsFromJSON() {
	auto systemJSON = json::parse(buffer.str());
	auto ArcsJSON = systemJSON.at("System").at("Arcs");
	unsigned arcGroupID;
	string predNodeGroupName, succNodeGroupName;
	string directedString, volumetricString;
	unsigned predID, succID, outDegree;
	value_t probExcitatory, weight;
	bool directed, volumetric;
	param_t Params;
	RandomArcModel arcGenerationModel;
	for (unsigned iarc = 0; iarc < ArcsJSON.size(); iarc++) {
		// unpack JSON
		arcGroupID = iarc;
		predNodeGroupName = ArcsJSON[iarc]["Source"].get<string>();
		predID = nodeGroupID[predNodeGroupName];
		succNodeGroupName = ArcsJSON[iarc]["Target"].get<string>();
		succID = nodeGroupID[succNodeGroupName];
		probExcitatory = ArcsJSON[iarc]["OutDegree"].get<value_t>();
		outDegree = ArcsJSON[iarc]["OutDegree"].get<unsigned>();
		string directedString = ArcsJSON[iarc]["Type"].get<string>();
		if (directedString == "directed") directed = true; else directed = false;
		string volumetricString = ArcsJSON[iarc]["Volumetric"].get<string>();
		if (volumetricString == "no") volumetric = false; else volumetric = true;
		arcGenerationModel = degreeModel;
		weight = ArcsJSON[iarc]["Weight"].get<value_t>();

		// create arcGroup structures
		string arcGroupName = predNodeGroupName + "->" + succNodeGroupName;
		cout << "Building Arc Groups with Name: " << arcGroupName << endl;
		//string Name, unsigned ID, state_t Params, vector<string> StateLabels, Environment* Env
		//state_t Params(2, 0.0);
		Params["gammaEp"] = 0.9;
		Params["gammaEm"] = 0.9;
		Params[ "gammaSTDP" ] = 0.99;
		Params["etaW"] = 0.01;
		vector<string> StateLabels{ "Force", "Offset" };
		Learner1* pLearner = new Learner1("Learner1", 0, Params, StateLabels, pEnv);
		switch (arcGroupID) {
		case AgentInput2AgentLayer1 :
		{
			Agent2Agent *pArcGroup = new Agent2Agent(arcGroupName, arcGroupID, predID, succID,
				probExcitatory, outDegree, directed, volumetric,
				arcGenerationModel, weight, pEnv, pLearner);
			pArcGroup->Init(*nodeGroups[predID], *nodeGroups[succID]);
			arcGroups.emplace_back(pArcGroup);
			break;
		}
		case AgentLayer12AgentLayer2 :
		{
			Agent2Agent *pArcGroup = new Agent2Agent(arcGroupName, arcGroupID, predID, succID,
				probExcitatory, outDegree, directed, volumetric,
				arcGenerationModel, weight, pEnv, pLearner);
			pArcGroup->Init(*nodeGroups[predID], *nodeGroups[succID]);
			arcGroups.emplace_back(pArcGroup);
			break;
		}
		case AgentLayer22AgentOutput:
		{
			Agent2Agent *pArcGroup = new Agent2Agent(arcGroupName, arcGroupID, predID, succID,
				probExcitatory, outDegree, directed, volumetric,
				arcGenerationModel, weight, pEnv, pLearner);
			pArcGroup->Init(*nodeGroups[predID], *nodeGroups[succID]);
			arcGroups.emplace_back(pArcGroup);
			break;
		}
		case AgentOutput2AgentAction:
		{
			AgOut2AgAct *pArcGroup = new AgOut2AgAct(arcGroupName, arcGroupID, predID, succID,
				probExcitatory, outDegree, directed, volumetric,
				arcGenerationModel, weight);
			pArcGroup->Init(*nodeGroups[predID], *nodeGroups[succID]);
			arcGroups.emplace_back(pArcGroup);
			break;
		}
		default:
		{
			BaseArcGroup* pArcGroup = new BaseArcGroup(arcGroupName, arcGroupID, predID, succID,
				probExcitatory, outDegree, directed, volumetric, arcGenerationModel, weight);
			pArcGroup->Init(*nodeGroups[predID], *nodeGroups[succID]);
			arcGroups.emplace_back(pArcGroup);
		}
		}

	}
}

void System::Init() {
	value_t *arcData, *nodeData;
	unsigned predID, succID;
	// Arcs
	for (int arcGrpID = 0; arcGrpID < arcGroups.size(); arcGrpID++) {
		predID = arcGroups[arcGrpID]->mPredID;
		succID = arcGroups[arcGrpID]->mSuccID;
		arcData = arcGroups[arcGrpID]->mStates.data();
		arcGroups[arcGrpID]->Init(*nodeGroups[predID], *nodeGroups[succID]);
	}
	// Nodes
	for (int nodeGrpID = 0; nodeGrpID < nodeGroups.size(); nodeGrpID++) {
		nodeData = nodeGroups[nodeGrpID]->mStates.data();
		nodeGroups[nodeGrpID]->Init();
	}
}

void System::Reset() {
	value_t *arcData, *nodeData;
	unsigned predID, succID;
	// Arcs
	for (int arcGrpID = 0; arcGrpID < arcGroups.size(); arcGrpID++) {
		predID = arcGroups[arcGrpID]->mPredID;
		succID = arcGroups[arcGrpID]->mSuccID;
		arcData = arcGroups[arcGrpID]->mStates.data();
		arcGroups[arcGrpID]->Reset(arcData);
	}
	// Nodes
	for (int nodeGrpID = 0; nodeGrpID < nodeGroups.size(); nodeGrpID++) {
		nodeData = nodeGroups[nodeGrpID]->mStates.data();
		nodeGroups[nodeGrpID]->Reset(nodeData);
	}
}

void System::Step(value_t t, value_t dt) {
	value_t *arcData, *nodeData;
	unsigned predID, succID;

	// Arcs
	for (int arcGrpID = 0; arcGrpID < arcGroups.size(); arcGrpID++) {
		predID = arcGroups[arcGrpID]->mPredID;
		succID = arcGroups[arcGrpID]->mSuccID;
		arcData = arcGroups[arcGrpID]->mStates.data();
		arcGroups[arcGrpID]->Step(arcData,
			*nodeGroups[predID], *nodeGroups[succID], t, dt);
	}
	// Nodes
	for (int nodeGrpID = 0; nodeGrpID < nodeGroups.size(); nodeGrpID++) {
		nodeData = nodeGroups[nodeGrpID]->mStates.data();
		nodeGroups[nodeGrpID]->Step(nodeData, t, dt);
	}
}

void System::Print(ofstream &ofs, bool printHeader) {
	value_t *arcData, *nodeData;
	unsigned predID, succID;

	if (printHeader) {
		value_t * data{ 0 };
		for (int nodeGrpID = 0; nodeGrpID < nodeGroups.size(); nodeGrpID++) {
			nodeGroups[nodeGrpID]->Print(data, ofs, true);
		}
		for (int arcGrpID = 0; arcGrpID < arcGroups.size(); arcGrpID++) {
			arcGroups[arcGrpID]->Print(data, ofs, true);
		}
	}
	else {
		for (int nodeGrpID = 0; nodeGrpID < nodeGroups.size(); nodeGrpID++) {
			nodeData = nodeGroups[nodeGrpID]->mStates.data();
			nodeGroups[nodeGrpID]->Print(nodeData, ofs, false);
		}
		for (int arcGrpID = 0; arcGrpID < arcGroups.size(); arcGrpID++) {
			arcData = arcGroups[arcGrpID]->mStates.data();
			arcGroups[arcGrpID]->Print(arcData, ofs, false);
		}
	}
}