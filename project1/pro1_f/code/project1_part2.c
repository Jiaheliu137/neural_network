#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N             3  // number of input neurons
#define R             3  // number of output neurons
#define n_sample      3
#define eta           0.5
#define lambda        1.0
#define desired_error 0.1
#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
#define step(x)       (x >= 0 ? 1 : -1)
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))
#define activation_function   0   // 0 for sigmoid, 1 for step

double x[n_sample][N]={
  {10,2,-1},
  {2,-5,-1},
  {-5,5,-1},
};
double d[n_sample][R]={
  {1,-1,-1},
  {-1,1,-1},
  {-1,-1,1},
};
double w[R][N];
double o[R];



void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main(){
  int    i,j,p,q=0;// q is the number of learning cycles, p is the number of samples, j is the number of output neurons, i is the number of input neurons
  double Error=DBL_MAX;
  double delta;

  Initialization();
  while(Error>desired_error){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
      FindOutput(p);
      for(i=0;i<R;i++){
        Error+=0.5*pow(d[p][i]-o[i],2.0);
      }
      for(i=0;i<R;i++){
        if (activation_function == 0) { // use Sigmoid function
          delta=(d[p][i]-o[i])*(1-o[i]*o[i])/2;
        } else if (activation_function == 1) { // use Step function
          delta = d[p][i] - o[i];
        }
	for(j=0;j<N;j++){
	  w[i][j]+=eta*delta*x[p][j];
	}
      }
    } 
    printf("Error in the %d-th learning cycle=%f\n",q,Error);
  }
  PrintResult();
  return 0;
}

void Initialization(void) {
  int i, j;

  randomize();
  for (i = 0; i < R; i++)
    for (j = 0; j < N; j++)
      w[i][j] = frand() - 0.5;
}

void FindOutput(int p) {
  int i, j;
  double temp;

  for (i = 0; i < R; i++) {
    temp = 0;
    for (j = 0; j < N; j++) {
      temp += w[i][j] * x[p][j];
    }
    if (activation_function == 0) {
      o[i] = sigmoid(temp);
    } else if (activation_function == 1) {
      o[i] = step(temp);
    }
  }
}

void PrintResult(void) {
  int i, j;

  printf("\n\n");
  printf("The connection weights are:\n");
  for (i = 0; i < R; i++) {
    for (j = 0; j < N; j++)
      printf("%5f ", w[i][j]);
    printf("\n");
  }
  printf("\n\n");
}

