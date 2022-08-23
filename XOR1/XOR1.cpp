
#include "pch.h"
#include "System.h"
#include "BaseNodeGroup.h"
#include "BaseArcGroup.h"
#include "BaseEnvironment.h"
#include "Environment.h"
#include "EnvOutput.h"

void printDataHeader(ofstream& ofs, string problem, string systemName, string device) {
	clock_t tm;
	tm = clock();
	time_t date_time = time(NULL);
	char str[26];
	ctime_s(str, sizeof str, &date_time);
	string dataHeader = problem + " " + systemName + "  Device:  " + device;
	dataHeader = dataHeader + "   " + str;
	ofs << dataHeader << "Trial,T,";
}
int main()
{
	// File Specifications
	const string WorkingDirectory = "A:\\Projects\\Dynamics\\";
	const string ConfigFileName = "XOR1.json";
	const string ConfigFileSpec = WorkingDirectory + ConfigFileName;

	// Open Log File
	const string logFile("A:\\Projects\\Dynamics\\state.csv");
	ofstream ofs(logFile, ofstream::out);
	int logFile_OK = ofs.is_open();
	bool log_output;
	unsigned LogPeriod = 1;			// Steps between outputs


	// Timer
	clock_t tm;

	// Parse JSON File and Build System
	System Sys(ConfigFileSpec);
	Sys.Init();

	//// Data Header
	printDataHeader(ofs, "XOR1", Sys.systemName, "CPU");

	// Label Header
	value_t * data{ 0 };
	for (int nodeGrpID = 0; nodeGrpID < Sys.nodeGroups.size(); nodeGrpID++) {
		Sys.nodeGroups[nodeGrpID]->print(data, ofs, 0.0, true);
	}
	ofs << "\n"; 	

	//// Run Simulation
	value_t dt = Sys.Eval->dt;
	value_t *arcData, *predNodeData, *succNodeData, *nodeData;
	unsigned predGroupID, succGroupID;
	//// Simulation over trials
	for (unsigned itrial = 0; itrial < Sys.Eval->numTrialsTraining; itrial++) {

		//// Simulation over time
		for (unsigned t = 0; t < Sys.Eval->numStepsTraining; t++) {
			std::cout << "Trial:  " << itrial << "  t:  " << t << '\n';
			// Arcs
			for (int arcGrpID = 0; arcGrpID < Sys.arcGroups.size(); arcGrpID++) {
				arcData = Sys.arcGroups[arcGrpID]->arcVector.data();
				predGroupID = Sys.arcGroups[arcGrpID]->predGroupID;
				predNodeData = Sys.nodeGroups[predGroupID]->nodeVector.data();
				succGroupID = Sys.arcGroups[arcGrpID]->succGroupID;
				succNodeData = Sys.nodeGroups[succGroupID]->nodeVector.data();
				Sys.arcGroups[arcGrpID]->Step_CPU(arcData, predNodeData, succNodeData, t, dt);
			}
			// Nodes
			for (int nodeGrpID = 0; nodeGrpID < Sys.nodeGroups.size(); nodeGrpID++) {
				nodeData = Sys.nodeGroups[nodeGrpID]->nodeVector.data();
				Sys.nodeGroups[nodeGrpID]->Step_CPU(nodeData, t, dt);
			}

			if (t%LogPeriod == 0) {
				ofs << itrial << "," << t;
				for (int nodeGrpID = 0; nodeGrpID < Sys.nodeGroups.size(); nodeGrpID++) {
					nodeData = Sys.nodeGroups[nodeGrpID]->nodeVector.data();
					Sys.nodeGroups[nodeGrpID]->print(nodeData, ofs, t, false);
				}
				ofs << '\n';
			}
		}
	}
	
}
