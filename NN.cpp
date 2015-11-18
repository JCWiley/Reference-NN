#include "NN.h"


NN::NN(long idum):seed(idum)
{}
int NN::get_num_input()
{
	return num_input;
}
int NN::get_num_hidden()
{
	return num_hidden;
}
int NN::get_num_output()
{
	return num_output;
}
double NN::get_learning_rate()
{
	return learning_rate;
}
//***************************************************
//creates a NN with all weights randomized
void NN::Create_NN(double rate,int input,int hidden,int output,double inc_momentum)
{
	num_input = input;
	num_hidden = hidden;
	num_output = output;

	momentum = inc_momentum;

	//resize variables to sizes provided by user
	input_values.resize(num_input);
	hidden_values.resize(num_hidden);
	output_values.resize(num_output);

	output_error.resize(num_output);

	target.resize(num_output);

	Hidden_Threshold.resize(num_hidden);
	Output_Threshold.resize(num_output);

	hidden_direction.resize(num_hidden);
	output_direction.resize(num_output);

	hidden_sum_inputs.resize(num_hidden);
	output_sum_inputs.resize(num_output);

	input_to_hidden_weights.resize(num_input);
	input_to_hidden_weight_changes.resize(num_input);

	hidden_to_output_weights.resize(num_hidden);
	hidden_to_output_weight_changes.resize(num_hidden);

	//initialize variables
	learning_rate = rate;
	for(int i = 0;i < num_input;i++)
	{
		input_values[i] = 0;
		input_to_hidden_weights[i].resize(num_hidden);
		input_to_hidden_weight_changes[i].resize(num_hidden);
	}
	for(int h = 0;h < num_hidden;h++)
	{
		hidden_values[h] = 0;
		Hidden_Threshold[h] = 0;
		hidden_to_output_weights[h].resize(num_output);
		hidden_to_output_weight_changes[h].resize(num_output);
		hidden_direction[h] = 0;
		hidden_sum_inputs[h] = 0;
	}
	for(int o = 0;o < num_output;o++)
	{
		output_values[o] = 0;
		output_error[o] = 0;
		target[o] = 0;
		Output_Threshold[o] = 0;
		output_direction[o] = 0;
		output_sum_inputs[o] = 0;
	}
	Randomize_weights();
}
//input data set to be trained on
void NN::Input_Data_Set(vector<double> set)
{
	for(int i = 0;i < num_input;i++)
	{
		input_values[i] = set[i];
	}
}
// propogate forward to determine resultant values
void NN::Forward_Propogate()
{
	Layer_Forward(true);
	Layer_Forward(false);
}
// provides the results of the computations to the master program
vector<double> NN::Get_Outputs()
{
	return output_values;
}
// gets the error from the master program
void NN::Set_Error(vector<double> error)
{
	if(error.size() == num_output)
	{
		for(int o = 0;o < num_output;o++)
		{
			output_error[o] = error[o];
		}
	}
	else
	{
		throw "error out of bounds";
	}
}
// adjusts weights according to the error
void NN::Adjust_weights()
{
	double temp1 = 0;
	//back propagate
	for(int X = 0;X < num_output;X++)//calculate exit layer direction
	{
		output_direction[X] = 0;
		output_direction[X] = Deriv_Squashing_function(output_sum_inputs[X]) * output_error[X];
	}
	for(int h = 0;h < num_hidden;h++)//calculate hidden layer direction
	{
		temp1 = 0;
		hidden_direction[h] = Deriv_Squashing_function(hidden_sum_inputs[h]);
		for(int o = 0;o < num_output;o++)
		{
			temp1 += output_direction[o] * hidden_to_output_weights[h][0];
		}
		hidden_direction[h] = hidden_direction[h] * temp1;
	}

	for(int h = 0;h < num_hidden;h++) //adjust layer 2 to 3 weights
	{
		for(int o = 0;o < num_output; o++)
		{
			hidden_to_output_weight_changes[h][o] = learning_rate * output_direction[o] * hidden_values[h] + (hidden_to_output_weight_changes[h][o] * momentum);
			hidden_to_output_weights[h][o]+= hidden_to_output_weight_changes[h][o];
		}
	}
	for(int o = 0;o < num_output;o++)//adjust layer 3 threshold weights
	{
		Output_Threshold[o] += learning_rate * output_direction[o] * 1.0;
	}


	for(int i = 0;i < num_input; i++) //adjust layer 1 to 2 weights
	{
		for(int h = 0;h < num_hidden;h++)
		{
			input_to_hidden_weight_changes[i][h] = learning_rate * hidden_direction[h] * input_values[i] + (input_to_hidden_weight_changes[i][h] * momentum);
			input_to_hidden_weights[i][h] += input_to_hidden_weight_changes[i][h];
		}
	}
	for(int H = 0;H < num_hidden;H++) //adjust layer 2 threshold weights
	{
		Hidden_Threshold[H] += learning_rate * hidden_direction[H] * 1.0;
	}
}
// saves a NN to a file
void NN::Save_NN()
{
	char name[50];
	cout << "Please name the file (include the .txt)\n";
	cin >> name;
	ofstream file;
	file.open(name);

	file << num_input << "\n";
	file << num_hidden << "\n";
	file << num_output << "\n";

	file << "\n";

	file << learning_rate << "\n";
	file << momentum << "\n";

	file << "\n";
	for(int h = 0;h < num_hidden;h++)
	{
		file << Hidden_Threshold[h] << "\n";
	}
	file << "\n";
	for(int o = 0;o < num_output;o++)
	{
		file << Output_Threshold[o] << "\n";
	}
	file << "\n";
	for(int h = 0;h < num_hidden;h++)
	{
		for(int i = 0;i < num_input;i++)
		{
			file << input_to_hidden_weights[i][h] << "\n";
		}
		file << "-\n";
	}
	file << "\n";
	for(int o = 0; o < num_output;o++)
	{
		for(int h = 0; h < num_hidden;h++)
		{
			file << hidden_to_output_weights[h][o] << "\n";
		}
		file << "-\n";
	}
	file.close();
}
// loads a NN from a file
void NN::Load_NN()
{
	char name[50];
	num_input = 0;
	num_hidden = 0;
	num_output = 0;

	char bin;

	bool flag = false;

	vector<vector<double> > buffer;
	double small_buffer;

	cout << "Please enter the name of the file to open\n";
	cin  >> name;

	ifstream file;
	file.open(name);

	if(file.is_open())
	{
		file >> num_input;
		file >> num_hidden;
		file >> num_output;

		file >> learning_rate;
		file >> momentum;

		for(int h = 0;h < num_hidden;h++)
		{
			file >> Hidden_Threshold[h];
		}
		for(int o = 0;o < num_output;o++)
		{
			file >> Output_Threshold[o];
		}
		for(int h = 0;h < num_hidden;h++)
		{
			for(int i = 0;i < num_input;i++)
			{
				file >> input_to_hidden_weights[i][h];
			}
			file >> bin;
		}
		for(int o = 0; o < num_output;o++)
		{
			for(int h = 0; h < num_hidden;h++)
			{
				file >> hidden_to_output_weights[h][o];
			}
			file >> bin;
		}
	}
	else
	{
		cout << "Invalid file\n";
	}


}

	
//***************************************************
// using sigmoid function to squash atm
double NN::Squashing_function(double input)
{
	float exp_value;
	float return_value;

	/*** Exponential calculation ***/
	exp_value = exp((double) -input);

	/*** Final sigmoid value ***/
	return_value = 1 / (1 + exp_value);

	return return_value;
}
// derivitive of the squashing funtion (deriv of sigmoid)
double NN::Deriv_Squashing_function(double input)
{
	return Squashing_function(input)*(1-Squashing_function(input));
}
// sets all weights to random values
void NN::Randomize_weights()
{
	for(int i = 0;i < num_input;i++)
	{
		for(int h = 0; h < num_hidden;h++)
		{
			input_to_hidden_weights[i][h] = uniform(&seed);
			if (uniform(&seed) > .5)
			{
				input_to_hidden_weights[i][h] = input_to_hidden_weights[i][h]*-1;
			}
		}
	}
	for(int h = 0;h < num_hidden;h++)
	{
		for(int o = 0;o < num_output; o++)
		{
			hidden_to_output_weights[h][o] = uniform(&seed);
			if (uniform(&seed) > .5)
			{
				hidden_to_output_weights[h][o] = hidden_to_output_weights[h][o]*-1;
			}
		}
	}
}
// resets all weights to 0
void NN::Zero_weights()
{
	for(int i = 0;i < num_input;i++)
	{
		for(int h = 0; h < num_hidden;h++)
		{
			input_to_hidden_weights[i][h] = 0;
		}
	}
	for(int h = 0;h < num_hidden;h++)
	{
		for(int o = 0;o < num_output; o++)
		{
			hidden_to_output_weights[h][o] = 0;
		}
	}
}
// Propogate a single layers values to the next layer 
// true for input to hidden, false for hidden to output
void NN::Layer_Forward(bool Layer)
{
	if(Layer == true)
	{
		for(int h = 0;h < num_hidden;h++)
		{
			hidden_values[h] = 0;
			hidden_sum_inputs[h] = 0;

			for(int i = 0;i < num_input;i++)
			{
				hidden_sum_inputs[h] += input_values[i] * input_to_hidden_weights[i][h];
			}
			hidden_sum_inputs[h] += Hidden_Threshold[h] * 1.0;
			hidden_values[h] = Squashing_function(hidden_sum_inputs[h]);
		}
	}
	else
	{
		for(int o = 0; o < num_output;o++)
		{
			output_values[o] = 0;
			output_sum_inputs[o] = 0;

			for(int h = 0;h < num_hidden;h++)
			{
				output_sum_inputs[o] += hidden_values[h] * hidden_to_output_weights[h][o];
			}
			output_sum_inputs[o] += Output_Threshold[o] * 1.0;
			output_values[o] = Squashing_function(output_sum_inputs[o]);
		}
	}
}
