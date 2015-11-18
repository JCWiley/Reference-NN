#include "NN.h"

#include <iostream>
#include <fstream>

void train(NN & network);
void train_from_txt(NN & network);
void test(NN & network);
void Initialize_NN(NN & network);

void main()
{
	long idum;
	int choice = 0;
	srand((unsigned) time(NULL));
	idum = -rand();
	bool done = false;

	NN * Test_network = new NN(idum);
	Initialize_NN(*Test_network);

	do
	{
		cout << "Please enter an option\n";
		cout << "1. Train Network with manual values\n";
		cout << "2. Train Network with values from a txt document\n";
		cout << "3. Test Network\n";
		cout << "4. Save Network\n";
		cout << "5. Load Network\n";
		cout << "6. Reinitialize Network\n";
		cout << "7. Exit\n";
		cin  >> choice;
		switch (choice)
		{
		case 1:
			train(*Test_network);
			break;
		case 2:
			train_from_txt(*Test_network);
			break;
		case 3:
			test(*Test_network);
			break;
		case 4:
			Test_network->Save_NN();
			break;
		case 5:
			Test_network->Load_NN();
			break;
		case 6:
			Initialize_NN(*Test_network);
			break;
		case 7:
			done = true;
			break;
		default:
			cout << "Please choose a different option\n";
			break;
		}
	}
	while(!done);

	delete Test_network;
}

void train_from_txt(NN & network)
{
	int num_test_vectors;
	int num_trials;
	int output;
	int input;

	char filename[100];

	vector<vector<double> > set;
	vector<vector<double> > target;
	vector<double> output_list;
	vector<double> error_list;

	output = network.get_num_output();
	input = network.get_num_input();

	cout << "Please enter the number of test vectors\n";
	cin  >> num_test_vectors;
	cout << "Please enter the number of trials\n";
	cin  >> num_trials;
	cout << "Please enter the filename to input from\n";
	cin  >> filename;

	set.resize(num_test_vectors);
	target.resize(num_test_vectors);
	output_list.resize(output);
	error_list.resize(output);

	ifstream file(filename);
	if(file.is_open())
	{
		for(int n = 0;n < num_test_vectors;n++)
		{
			//input testing variables
			set[n].resize(input);
			target[n].resize(output);
			//input the values to learn
			for(int i = 0; i < input;i++)
			{
				file >> set[n][i];
			}
			//intput the results to compare against for current set
			for(int o = 0; o < output;o++)
			{
				file >> target[n][o];
			}
		}

		cout << "Processing\n";
		for(int i = 0;i < num_trials;i++)
		{
			for(int n = 0;n < num_test_vectors;n++)
			{
				//train network
				network.Input_Data_Set(set[n]);
				network.Forward_Propogate();
				output_list = network.Get_Outputs();
				for(int o = 0;o < output;o++)
				{
					error_list[o] = target[n][o] - output_list[o];
				}
				network.Set_Error(error_list);
				network.Adjust_weights();
			}
		}
	}
	else cout << "Unable to open file"; 
}
void train(NN & network)
{
	int num_sample_inputs;
	int num_trials;
	int output;
	int input;

	vector<vector<double> > set;
	vector<vector<double> > target;
	vector<double> output_list;
	vector<double> error_list;

	output = network.get_num_output();
	input = network.get_num_input();

	cout << "Please enter the number of test vectors\n";
	cin  >> num_sample_inputs;
	cout << "Please enter the number of trials\n";
	cin  >> num_trials;

	set.resize(num_sample_inputs);
	target.resize(num_sample_inputs);
	output_list.resize(output);
	error_list.resize(output);

	for(int n = 0;n < num_sample_inputs;n++)
	{
		//input testing variables
		set[n].resize(input);
		target[n].resize(output);
		cout << "\nPlease enter "<< input << " numbers\n";

		for(int i = 0; i < input;i++)
		{
			cin >> set[n][i];
		}

		cout << "Please enter the desired result\n";
		for(int o = 0; o < output;o++)
		{
			cin >> target[n][o];
		}
	}
	cout << "Processing\n";
	for(int i = 0;i < num_trials;i++)
	{
		for(int n = 0;n < num_sample_inputs;n++)
		{
			//train network
			network.Input_Data_Set(set[n]);
			network.Forward_Propogate();
			output_list = network.Get_Outputs();
			for(int o = 0;o < output;o++)
			{
				error_list[o] = target[n][o] - output_list[o];
			}
			network.Set_Error(error_list);
			network.Adjust_weights();
		}
	}
}
void test(NN & network)
{
	char cont;

	int num_input = network.get_num_input();
	int num_hidden = network.get_num_hidden();
	int num_output = network.get_num_output();

	vector<double> test;
	vector<double> result;

	test.resize(num_input);
	result.resize(num_output);

	do
	{
		cout << "Please enter a test vector for the network\n";
		for(int i = 0; i < num_input;i++)
		{
			cin >> test[i];
		}

		network.Input_Data_Set(test);
		network.Forward_Propogate();
		result = network.Get_Outputs();

		for(int o = 0;o < num_output;o++)
		{
			cout << result[o] << "\n";
		}

		cout << "Do you want to test another set of values? (y/n) \n";
		cin >> cont;
	}
	while (cont != 'n');
}
void Initialize_NN(NN & network)
{
	int input;
	int hidden;
	int output;
	double learning_rate;
	double momentum;
	//input NN specifications
	cout << "Please enter the number of entry nodes\n";
	cin  >> input;
	cout << "Please enter the number of hidden nodes (one layer)\n";
	cin  >> hidden;
	cout << "Please enter the number of exit nodes\n";
	cin  >> output;
	cout << "Please enter the learning rate (0 < learning rate < 1)\n";
	cin  >> learning_rate;
	cout << "Please enter a momentum value for the network (0 < momentum < .5)\n";
	cin  >> momentum;
	network.Create_NN(learning_rate,input,hidden,output,momentum);
}