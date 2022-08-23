#pragma once
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

using namespace std;
using json = nlohmann::json;


// typedefs
typedef float value_t;
typedef vector<value_t> state_t;
typedef vector<unsigned> ustate_t;
typedef vector<int> intstate_t;
typedef vector<vector<value_t>> array_t;

// random
default_random_engine rng;
uniform_real_distribution<value_t> unifdistribution(0.0, 1.0);
normal_distribution<value_t> normaldistribution(0.0, 1.0);

// enums
enum RandomArcModel { probModel, degreeModel };
enum NodeGroupNameEnum {
	AgentInput,
	AgentLayer1,
	AgentLayer2,
	AgentOutput,
	AgentAction,
	EnvironmentHeat,
	EnvironmentWork,
	EnvironmentOutput,
	EnvironmentReward
};