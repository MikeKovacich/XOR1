#pragma once
#include "pch.h"
#include "BaseNodeGroup.h"
#include "BaseArcGroup.h"
#include "Environment.h"
#include "Evaluation.h"
#include "EnvOutput.h"

struct System{

	string systemJSONFile;
	stringstream buffer;
	string systemName;
	Environment* Env;
	Evaluation* Eval;
	vector<unique_ptr<BaseNodeGroup>> nodeGroups;
	vector<unique_ptr<BaseArcGroup>> arcGroups;
	map<string, unsigned> nodeGroupID;

	// ctor
	System(string systemJSONFile_) : systemJSONFile(systemJSONFile_) {};

	// copy constructor
	System(const System &Sys);

	void Init();

	void readSystemJSON();
	void buildEnvironmentFromJSON(unsigned numStepsPerCase);
	void buildEvaluationFromJSON();
	void buildNodesFromJSON();
	void buildArcsFromJSON();

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

void System::Init() {
	readSystemJSON();
	buildEvaluationFromJSON();
	buildEnvironmentFromJSON(Eval->numStepsPerCase);
	buildNodesFromJSON();
	buildArcsFromJSON();
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

	Eval = new Evaluation(Label, numStepsTraining, numTrialsTraining,
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
	Env = new Environment(Label, numStepsPerCase, numCases, randomCases);
}


void System::buildNodesFromJSON() {
	auto systemJSON = json::parse(buffer.str());
	auto NodesJSON = systemJSON.at("System").at("Nodes");
	
	string nameGroup;
	unsigned numNode;
	unsigned groupID;
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
			break;
		case 1:
			//AgentLayer1
			break;
		case 2:
			//AgentLayer2
			break;
		case 3:
			//AgentOutput
			break;
		case 4:
			//AgentAction
			break;
		case 5:
			// EnvironmentHeat
			break;
		case 6:
			// EnvironmentWork
			break;
		case 7:
			// EnvironmentOutput
		{
			EnvOutput *NodeGroupPtr = new EnvOutput(nameGroup, groupID, numNode, Env);
			NodeGroupPtr->Init();
			nodeGroups.emplace_back(NodeGroupPtr);
			break;
		}	
		case 8:
			// EnvironmentReward
			break;
		default:
			BaseNodeGroup *NodeGroupPtr = new BaseNodeGroup(nameGroup, groupID, numNode);
			NodeGroupPtr->Init();
			nodeGroups.emplace_back(NodeGroupPtr);
		}


		
	}
}

void System::buildArcsFromJSON() {
	auto systemJSON = json::parse(buffer.str());
	auto ArcsJSON = systemJSON.at("System").at("Arcs");

	string predNodeGroupName, succNodeGroupName;
	string directedString, volumetricString;
	unsigned predID, succID, outDegree;
	value_t probExcitatory;
	bool directed, volumetric;
	RandomArcModel arcGenerationModel;
	for (int iarc = 0; iarc < ArcsJSON.size(); iarc++) {
		// unpack JSON
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

		// create arcGroup structures
		cout << "Building Arc with Source: " << predNodeGroupName << "  and Target: " << succNodeGroupName << endl;
		BaseArcGroup *arcGroupPtr = new BaseArcGroup(*nodeGroups[predID], predID, *nodeGroups[succID], succID,
			probExcitatory, outDegree, directed, volumetric, arcGenerationModel);
		arcGroupPtr->Init();

		// add to containers of nodeGroup structures	
		arcGroups.emplace_back(arcGroupPtr);
	}
}