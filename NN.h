#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "random.h"
#include <math.h>

using std::vector;
using std::cout;
using std::cin;
using std::ifstream;
using std::ofstream;

class NN
{
public:
	//********************Cannonical***********************
	NN(long seed);
		
	int get_num_input();
	int get_num_hidden();
	int get_num_output();

	double get_learning_rate();
	vector<double> Get_Outputs();

	//*********************Main*************************
	//creates a NN with all weights randomized
	void Create_NN(double rate,int input,int hidden,int output,double inc_momentum);
	//input data set to be trained on
	void Input_Data_Set(vector<double> set);
	// propagate forward to determine resultant values
	void Forward_Propogate();
	// gets the error from the master program
	void Set_Error(vector<double>);
	// adjusts weights according to the error
	void Adjust_weights();
	// saves the NN to a file // todo
	void Save_NN();
	// loads a NN from a file // todo
	void Load_NN();
	
private:
	//********************************************* Functions *****************************************//
	// sigmoid atm
	double Squashing_function(double input);
	//derivitive of sigmoid
	double Deriv_Squashing_function(double input);
	// sets all weights to random values
	void Randomize_weights();
	// resets all weights to 0
	void Zero_weights();
	// Propogate a single layers values to the next layer // true for input to hidden, false for hidden to output
	void Layer_Forward(bool Layer);


	//********************************************* Variables *****************************************
	//learning rate for neural network
	double learning_rate;

	//momentum value for network
	double momentum;

	//direction variables, represent the direction to travel on the curve to reach the proper answer
	vector<double> output_direction;
	vector<double> hidden_direction;

	//sum of inputs to nodes pre squashing function
	vector<double> output_sum_inputs;
	vector<double> hidden_sum_inputs;

	//the number of nodes in each layer of the network
	int num_input;
	int num_hidden;
	int num_output;

	//the values at each node for the current passthrough,
	vector<double> input_values;
	vector<double> hidden_values;
	vector<double> output_values;

	//error for hidden and output nodes
	vector<double> output_error;

	//array/vector of target output variables
	vector<double> target;

	//weights
	vector<vector<double> > input_to_hidden_weights;
	vector<vector<double> > hidden_to_output_weights;
	
	//previous cycle weight changes (for momentum calulations)
	vector<vector<double> > input_to_hidden_weight_changes;
	vector<vector<double> > hidden_to_output_weight_changes;

	// threshold values for each layer
	vector<double > Hidden_Threshold;
	vector<double > Output_Threshold;

	//seed for random number generation
	long seed;
};