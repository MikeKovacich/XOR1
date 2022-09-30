
#include "pch.h"
#include "System.h"
#include "BaseGroup.h"
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
	ofs << dataHeader << "Trial,T";
}
int main()
{

	// File Specifications
	const string WorkingDirectory = "A:\\Projects\\Dynamics\\";
	const string ConfigFileName = "XOR1.json";
	const string ConfigFileSpec = WorkingDirectory + ConfigFileName;

	// Open Log File
	const string logFile("A:\\Projects\\Dynamics\\XOR\\data\\state.csv");
	ofstream ofs(logFile, ofstream::out);
	int logFile_OK = ofs.is_open();
	bool log_output;
	unsigned LogPeriod = 1;			// Steps between outputs

	// Open Graph File (GDF format)
	const string graphFileSpec("A:\\Projects\\Dynamics\\graph.gdf");
	

	// Timer
	clock_t tm;

	// Parse JSON File and Build System
	System Sys(ConfigFileSpec);
	Sys.Build();
	Sys.PrintGraph(graphFileSpec);

	//// Data Header
	printDataHeader(ofs, "XOR1", Sys.systemName, "CPU");

	// Label Header
	Sys.Print(ofs, true);
	ofs << "\n"; 	

	//// Run Simulation
	value_t dt = Sys.pEval->dt;
	//// Simulation over trials
	for (unsigned itrial = 0; itrial < Sys.pEval->numTrialsTraining; itrial++) {
		if (itrial == 0) Sys.Init(); else Sys.Reset();
		//// Simulation over time
		for (unsigned t = 0; t < Sys.pEval->numStepsTraining; t++) {
			std::cout << "Trial:  " << itrial << "  t:  " << t << '\n';
			Sys.Step(t, dt);
			if (t%LogPeriod == 0) {
				ofs << itrial << "," << t;
				Sys.Print(ofs, false);
				ofs << '\n';
			}
		}
	}
	
}
