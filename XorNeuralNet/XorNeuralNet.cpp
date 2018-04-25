#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include "include\fann.h"
#include "include\floatfann.h"

#define SIG(x) (1.0/(1.0+exp((-1.0)*x)))		// сигмоидальная функция активации
#define SIG_D(x) (exp((-1.0)*x)/((1.0+exp((-1.0)*x))*(1.0+exp((-1.0)*x))))  // деривация сигмоида
#define RAND_R(m) (rand()%m)                // генератор целого от 0 до m
#define RAND_D ((double)rand()/(double)RAND_MAX)   // генератор double от 0 до 1
#define SIZE 2

using namespace std;

typedef struct struct_dataset {   // обучающая структура
	double inputs[2];     // входные агрументы xor
	double result;        // возможный результат 0 или 1
} type_dataset;

type_dataset arr_xor_operation[] = { { { 0, 0 }, 0 },{ { 0, 1 }, 1 },{ { 1, 0 }, 1 },{ { 1, 1 }, 0 }, };   // массив таблицы истинности для xor

typedef struct str_type_neuron {    // структура нейрона
	double inputs[SIZE];   // входы
	double w_neur[SIZE];  // вес каждого входа
	double b;             // смещение
} type_neuron;

void n_init(type_neuron *a_neur) {      // рандомная инициализация
	for (size_t i = 0; i < SIZE; i++)
		a_neur->w_neur[i] = RAND_D;
	a_neur->b = RAND_D;
}

double run_forw(type_neuron *a_neur) {        // вычисление
	double ret = 0;
	for (size_t i = 0; i < SIZE; i++)
		ret += a_neur->inputs[i] * a_neur->w_neur[i];
	ret += a_neur->b;
	return SIG(ret);
}

void run_backw(type_neuron *a_neur, double error) {   // обновляем веса нейронов и смещение
	for (size_t i = 0; i < SIZE; i++)
		a_neur->w_neur[i] += error * a_neur->inputs[i];
	a_neur->b += error;
}


int main() {
	/*	
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
	*/

	type_neuron t_hid_neur[SIZE];                     // определяем hidden-слой нейросети
	for (size_t i = 0; i < SIZE; i++)
		n_init(&t_hid_neur[i]);
	type_neuron t_out_neur;                           // опр. выходной слой
	n_init(&t_out_neur);

	for (size_t i = 0; i < 1000; i++) {              // счётчик эпох
		type_dataset t_data = arr_xor_operation[RAND_R(4)];
		printf("\nInput        Output       Des.out.     Error:\n");
		printf("-------------------------------------------------\n");
		printf("%d ^ %d ", (int)t_data.inputs[0], (int)t_data.inputs[1]);
		for (size_t i = 0; i < SIZE; i++) {
			type_neuron *ptr = &t_hid_neur[i];
			for (size_t j = 0; j < SIZE; j++)
				ptr->inputs[j] = t_data.inputs[j];
			t_out_neur.inputs[i] = run_forw(ptr);
		}
		double out = run_forw(&t_out_neur);
		double out_err = SIG_D(out) * (t_data.result - out);
		run_backw(&t_out_neur, out_err);            // метод обратного распространения ошибки
		for (size_t i = 0; i < SIZE; i++) {
			double err = SIG_D(t_out_neur.inputs[i]) * out_err * t_out_neur.w_neur[i];
			run_backw(&t_hid_neur[i], err);
		}
		printf("result %d (%d)       %f       %f\n", out>0.2, (int)t_data.result, out, out_err);
	}
	return 0;
}
