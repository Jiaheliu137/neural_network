#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I             3
#define n_sample      4
#define eta           0.5//learning rate
#define lambda        1.0
#define desired_error 0.01
#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
#define step(x)       (x >= 0 ? 1 : -1)
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))//randomize the seed of the random number generator
#define activation_function   1   // 0 for sigmoid, 1 for step

double x[n_sample][I]={
  { 0, 0,-1},
  { 0, 1,-1},
  { 1, 0,-1},
  { 1,1,-1},
};

double w[I];
double d[n_sample]={-1,-1,-1,1};
double o;//the actual output

double record[n_sample]={0,0,0,0};

void Initialization(void); // Initialization of the connection weights
void FindOutput(int); // Find the actual outputs of the network
void PrintResult(void);
void printArray(double arr[], int size, int q);

int main(){
  int    i, p, q=0; // q is the number of learning cycles, p is the number of samples, i is the number of neurons
  double delta, Error=DBL_MAX; // Error=DBL_MAX is the maximum value of double

  Initialization();
  while(Error > desired_error){
    q++;
    Error = 0;
    for(p=0; p<n_sample; p++){
      FindOutput(p); // get the value of o
      record[p] = o;
      // printf("the out put of each pattern %d: %f\n",p,o);
      Error += 0.5*pow(d[p]-o, 2.0); // d[p] is the desired output, o is the actual output
      for(i=0; i<I; i++){
        // delta = (d[p]-o)*(1-o*o)/2;
        if (activation_function == 0) { // use Sigmoid function
          delta = (d[p]-o)*(1-o*o)/2;
        } else if (activation_function == 1) { // use Step function
          delta = d[p] - o;
        }
        w[i] += eta*delta*x[p][i];
      }
      // printf("Error in the %d-th learning cycle=%f\n",q,Error);
    } 
    printArray(record, n_sample, q);
  }
  PrintResult();

  return 0;
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i;

  randomize();
  for(i=0; i<I; i++) w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p){
  int i;
  double temp = 0;

  for(i=0; i<I; i++) temp += w[i]*x[p][i];
  // o = sigmoid(temp);
  if (activation_function == 0) o = sigmoid(temp);
  else if (activation_function == 1) o = step(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i;

  printf("\n\n");
  printf("The connection weights of the neurons:\n");
  for(i=0; i<I; i++) printf("%5f ", w[i]);
  printf("\n\n");
}

void printArray(double arr[], int size, int q){
  int i;
  printf("the out put of each pattern in %d-th learning cycle: ", q);
  for(i=0; i<size; i++){
    printf("%f ", arr[i]);
  }
  printf("\n");
}
