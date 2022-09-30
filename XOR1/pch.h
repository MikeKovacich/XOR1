#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <iterator>
#include <algorithm>
#include <map>
#include <string>
#include <memory>
#include <tuple>
#include "A:\Projects\JSON\json.hpp"
#include <time.h>
#include <array>
#include <vector>
#include <chrono>

using namespace std;
using json = nlohmann::json;


// typedefs
typedef float value_t;
typedef vector<value_t> state_t;
typedef vector<unsigned> ustate_t;
typedef vector<int> intstate_t;
typedef vector<vector<value_t>> array_t;
typedef map<string, double> param_t;

// random
default_random_engine rng;
uniform_real_distribution<value_t> unifdistribution(0.0, 1.0);
normal_distribution<value_t> normaldistribution(0.0, 1.0);

// enums
enum RandomArcModel { probModel, degreeModel };
enum NodeGroupNameEnum {
	AgentInputGroup,
	AgentLayer1Group,
	AgentLayer2Group,
	AgentOutputGroup,
	AgentActionGroup,
	EnvHeatGroup,
	EnvWorkGroup,
	EnvOutputGroup,
	EnvRewardGroup
};
enum ArcGroupNameEnum {
	EnvOutput2AgentInput,
	AgentInput2AgentLayer1,
	AgentLayer12AgentLayer2,
	AgentLayer22AgentOutput,
	AgentOutput2AgentAction,
	AgentAction2EnvReward,
	EnvReward2Agent,
	Agent2EnvHeat,
	Agent2EnvWork
};