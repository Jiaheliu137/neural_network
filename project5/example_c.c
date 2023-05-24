/*************************************************************/
/* C-program for self-organizing feature map                 */
/*                                                           */
/*  1) Number of input : I                                   */
/*  2) Number of neurons: N                                  */
/*  3) Number of training patterns: P                        */
/*                                                           */
/* The program is verified using the example (with a bit     */
/* change) given in Kohonen, "The self-organizing map"       */
/* IEEE Proceding, Vol. 78, No. 9, pp. 1469-1470             */
/*                                                           */
/* This program is produced by Qiangfu Zhao.                 */
/* You are free to use it for educational purpose            */
/*************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 5
#define P 32
#define M1 11
#define M2 11
#define N (M1 * M2)

#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

double x[P][I], y[N], w[N][I]; //
int label0[P];
char label[N];

void InputPattern(char *);
void Initialization(void);
void SOFM(int, double, double, int, int);
void Calibration(void);
void PrintResult(void);

/*************************************************************/
/* The main program                                          */
/* Usage: Command FileName                                   */
/*************************************************************/
int main(int argc, char **argv)
{

  if (argc != 2)
  {
    printf("Usage: %s FileName\n", argv[0]);
    printf("Please try again\n");
    exit(0);
  }

  Initialization();
  InputPattern(argv[1]);

  SOFM(1000, 0.5, 0.04, 10, 1);
  Calibration();
  printf("\n\nResult after the first 1,000 iterations:\n");
  PrintResult();

  SOFM(9000, 0.04, 0.0, 1, 1);
  Calibration();
  printf("\n\nResult after 10,000 iterations:\n");
  PrintResult();
}

/*************************************************************/
/* Input the patterns and their labels                       */
/*************************************************************/
void InputPattern(char filename[])
{
  int i, p;
  FILE *fp;
  int f[I];

  /* This part is used for input the example data */

  if ((fp = fopen(filename, "r")) == NULL)
  {
    printf("Can't open data file !\n");
    exit(1);
  }

  for (p = 0; p < P; p++)
  {
    for (i = 0; i < I; i++)
    {
      fscanf(fp, "%lf", &x[p][i]);
      printf("%5.0f\t", x[p][i]);
    }
    fscanf(fp, "%d", &label0[p]);
    printf("%d\n", label0[p]);
    printf("%c\n",
           (label0[p] <= 25) ? ('A' + label0[p]) : ('0' + label0[p] - 25));
  }
  fclose(fp);
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i, n;
  randomize();
  for (n = 0; n < N; n++)
  {
    for (i = 0; i < I; i++)
    {
      w[n][i] = frand();
    }
  }
}

/*************************************************************/
/* Self-organizing map                                       */
/*************************************************************/
void SOFM(int n_update, double r1, double r2, int Nc1, int Nc2)
{
  int i, p, q;
  int m1, m2, m10, m20;
  double d, d0;
  double x1, x2;
  double r;
  int nc;

  for (q = 0; q < n_update; q++)
  {

    /* Step1: Select a sample at random from the training set */

    p = rand() % P;

    /* Step2: Find the neuron closest to the input */

    d0 = DBL_MAX;
    for (m1 = 0; m1 < M1; m1++)
    {
      for (m2 = 0; m2 < M2; m2++)
      {
        d = 0;
        for (i = 0; i < I; i++)
          d += pow(w[m1 * M2 + m2][i] - x[p][i], 2.0);
        if (d < d0)
        {
          d0 = d;
          m10 = m1; 
          m20 = m2;
        }
      }
    }

    /* Step3: Update the weight of the neuron if it is in the neighborhood */
    /*        of the (m10,m20)-th neuron                                   */

    r = q * (r2 - r1) / n_update + r1; // r is learning rate
    nc = q * (Nc2 - Nc1) / n_update + Nc1; // nc is neighborhood size

    for (m1 = 0; m1 < M1; m1++)
    {
      x1 = m1;
      for (m2 = 0; m2 < M2; m2++)
      {
        ax.annotate(text, (x[i], y[i]), ha='center', va='center', color=color, fontsize=fontsize)
        x2 = m2;
        if (m1 % 2 == 0)
          x2 += 0.5; //+0.5 is used to make the hexagonal lattice
        d = sqrt(pow(x1 - m10, 2.0) + pow(x2 - m20, 2.0));
        if ((int)d <= nc)
        {
          for (i = 0; i < I; i++)
          {
            w[m1 * M2 + m2][i] += r * (x[p][i] - w[m1 * M2 + m2][i]);
          }
        }
      }
    }
  }
}

/*************************************************************/
/* Calibration                                               */
/*************************************************************/
void Calibration(void)
{
  int p, i, n, n0;
  double d, d0;

  for (n = 0; n < N; n++)
    label[n] = '*';

  for (p = 0; p < P; p++)
  {
    d0 = DBL_MAX;
    for (n = 0; n < N; n++)
    {
      d = 0;
      for (i = 0; i < I; i++)
      {
        d += pow(w[n][i] - x[p][i], 2.0);
      }
      if (d < d0)
      {
        d0 = d;
        n0 = n;
      }
    }
    label[n0] = (label0[p] <= 25) ? ('A' + label0[p]) : ('0' + label0[p] - 25);
  }
}

/*************************************************************/
/* Print out the result of the q-th iteration                */
/*************************************************************/
void PrintResult(void)
{
  int m1, m2;

  printf("\n\n");
  for (m1 = 0; m1 < M1; m1++)
  {
    if (m1 % 2 == 0)
      printf(" ");
    for (m2 = 0; m2 < M2; m2++)
    {
      printf("%c ", label[m1 * M2 + m2]);
    }
    printf("\n");
  }
  printf("\n\n");
}
