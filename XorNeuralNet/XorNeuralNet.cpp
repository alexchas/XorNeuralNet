#include "stdafx.h"
#include "stdio.h"
#include "include\fann.h"
#include "include\floatfann.h"

using namespace std;

int main() {
	int allLay = 3, inpNeur = 2, outNeur = 1, hidNeur = 2, aeons = 500, aeonsPause = 2;
	float infelicity = (float)0.0001;
	struct fann *neuralNet = fann_create_standard(allLay, inpNeur, hidNeur, outNeur);
	fann_type *outData;
	fann_type xor_arr[2] = { -1, 1 };

	fann_set_activation_function_hidden(neuralNet, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(neuralNet, FANN_SIGMOID_SYMMETRIC);
	fann_train_on_file(neuralNet, "input_data.txt", aeons, aeonsPause, infelicity);
	outData = fann_run(neuralNet, xor_arr);

	printf("Probability for input data: (%f, %f) is %f\n", xor_arr[0], xor_arr[1], outData[0]);
	fann_save(neuralNet, "out.net");
	fann_destroy(neuralNet);
	system("pause");
	return 0;
}
