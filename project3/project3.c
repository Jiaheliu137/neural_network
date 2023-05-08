/***********************************************************************/
/*We refer to the code for class. */
/***********************************************************************/
#include <stdlib.h> 
#include <stdio.h>
#include <time.h>

#define n_neuron   120 //12*10
#define n_pattern  4 //number of pattern (0,2,4,6)
#define n_row      10 
#define n_column   12 
#define noise_rate 0.15 //{0% 10% 15%}

int pattern[n_pattern][n_neuron]={
  {1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,  
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1},
  {1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
   1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,
   1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-
     1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1}};

int w[n_neuron][n_neuron];//120*120
int v[n_neuron]; //vector 120
/***********************************************************************/
/* Output the patterns, to confirm they are the desired ones           */
/***********************************************************************/
void Output_Pattern(int k) {
  FILE *outputfile; 
  int i,j;
  
  outputfile = fopen("./outpat3.txt", "a");
  fprintf(outputfile,"Pattern[%d]:\n",k);
  for(i=0;i<n_row;i++)
    { for(j=0;j<n_column;j++)
	fprintf(outputfile,"%2c",(pattern[k][i*n_column+j]==-1)?'*':' ');
      fprintf(outputfile,"\n");
    }
  fprintf(outputfile,"\n\n\n");
  fclose(outputfile);
  getchar();
}

/***********************************************************************/
/* Output the state of the network in the form of a picture            */
/***********************************************************************/
void Output_State(int k) {
  int i,j;
  FILE *outputfile; 
  
  outputfile = fopen("./outsta3.txt", "a");
  
  fprintf(outputfile, "%d-th iteration:\n",k);
  for(i=0;i<n_row;i++)
    { for(j=0;j<n_column;j++)
	fprintf(outputfile,"%2c",(v[i*n_column+j]==-1)?'*':' ');
      fprintf(outputfile,"\n");
    }
  fprintf(outputfile,"\n\n\n");
  fclose(outputfile);
  getchar();
}

/***********************************************************************/
/* Store the patterns into the network                                 */
/***********************************************************************/
void Store_Pattern() {
  //Phase1: Storage
  int i,j,k;
  for(i=0; i<n_neuron; i++){ 
    for(j=0; j<n_neuron; j++){ 
      w[i][j] = 0; //Initialization W =0
	  
      for(k=0;k<n_pattern;k++)
        w[i][j] += pattern[k][i]*pattern[k][j];//W=W+s(m)*s(m)^T
	   w[i][j] /= (double)n_pattern; //nomalazation
	 }
      w[i][i]=0;//w_ii=0
  }
}

/***********************************************************************/
/* Recall the m-th pattern from the network                            */
/* The pattern is corrupted by some noises                             */
/***********************************************************************/
void Recall_Pattern(int m){
  //Phase 2: Recall
  int   i,j,k;
  int   n_update;
  int   net, vnew;
  double r;

  for(i=0;i<n_neuron;i++){ 
    r=(double)(rand()%10001)/10000.0;//r:random
    if(r<noise_rate)
      v[i]=(pattern[m][i]==1)?-1:1;//corrupted by noises
    else
      v[i]=pattern[m][i]; //v<-pattern
  }
  Output_State(0);         /* show the noisy input pattern */

  k=1;
  do {//recall
    n_update=0;//
    for(i=0; i<n_neuron; i++){ 
      net=0;
      for(j=0; j<n_neuron; j++)//Find the new output
        net+=w[i][j]*v[j];//Σ(w*v)
	    if(net >= 0) vnew = 1; 
	    if(net < 0) vnew = -1;
	    if(vnew != v[i]) {
        n_update++;
        v[i]=vnew;
      }
    }
    Output_State(k);     /* show the current result */
    k++;
  } while(n_update!=0); //Stop if there is no change after many iterations
}

/***********************************************************************/
/* Initialize the weights                                              */
/***********************************************************************/
void Initialization()
{int i,j;

 for(i=0; i<n_neuron; i++)
   for(j=0; j<n_neuron; j++)
     w[i][j] = 0;
}

/***********************************************************************/
/* The main program                                                    */
/***********************************************************************/
int main(){ 
  int k;

  for(k=0;k<n_pattern;k++) Output_Pattern(k);

  Initialization();
  Store_Pattern();
  for(k=0;k<n_pattern;k++) Recall_Pattern(k);
}  
