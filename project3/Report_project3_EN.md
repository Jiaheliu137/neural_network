# Project3

## 1.Team member



- **m5268101 Liu** **Jiahe**

- **m5251140 Ken Sato**

- **m5271051 Keisuke** **Utsumi**



## 2.Team Project III

- Team Project III (cont.)
  - Write a computer program to implement the algorithm.
  - Try to recall the patterns with the noise level being 0%, 10% or 15%.
  - We say a pixel is a noise if its value is changed from 1 (or 0) to 0 (or 1).

## 3.Mathematical formulas

------

The main uses of Hopfield neural networks include two aspects: generating weight matrices and performing Recall on specified patterns.

### 1.Generate weight matrix

$$
w_{ij} = \frac{1}{P}\sum_{k=1}^{P} x_{i}^{(k)}x_{j}^{(k)} \quad \text{for} \quad i \neq j, x_{ii}=0
$$

$w_{ij}$ is the element in the $i$-th row and $j$-th column of the weight matrix, $N$ is the number of neurons (i.e., the number of elements in the pattern), $P$ is the number of patterns, and $x_{i}^{(k)}$ is the value of the $i$-th neuron in the $k$-th pattern.

Written in matrix form:
$$
W_{N \times N} = \frac{1}{P}\sum_{k=1}^{P} X^{(k)T}X^{k} \quad\text{for} \quad W_{i,i}= 0
$$
$X^{k}$ represents the k-th pattern, which is a row vector containing N neurons.

```python
for key in original_patterns:
    pattern = original_patterns[key]
    w += np.outer(pattern, pattern)
w /= n_pattern
np.fill_diagonal(w, 0)
```



### 2. Recall

The recall process of the Hopfield network can be represented as:
$$
x_{i}(t+1) = \mathrm{sgn}\left(\sum_{j=1}^{N} w_{ij}x_{j}(t)\right)
$$
$x_{i}(t)$ is the value of the $i$-th neuron at time step $t$, $\mathrm{sgn}$ is the sign function ($sgn(x) = 1 $ for $x \ge 0$; $sgn(x) = -1$ for $x < 0$), and $w_{ij}$ is the element in the $i$-th row and $j$-th column of the weight matrix.

There are two ways of updating: asynchronous and synchronous. Asynchronous updating updates one neuron at a time, while synchronous updating updates all neurons at once. One round is completed when all neurons are updated. The iteration stops when the neurons in the next round do not change compared to the previous round.

```python
while True:
        
        if recall_method == synchronous_update:
            new_pattern = np.where(w @ noisy_pattern >= 0, 1, -1)
            
            if np.array_equal(new_pattern, noisy_pattern):
                break
            
            noisy_pattern = new_pattern
        
        if recall_method == asynchronous_update:

            prev_pattern = noisy_pattern.copy()

            for i in range(n_neuron):
                net_input = w[i] @ noisy_pattern

                noisy_pattern[i] = 1 if net_input >= 0 else -1

            if np.array_equal(noisy_pattern, prev_pattern):
                break
```

## 4.Implement code

There is an existing [pattern](pattern.json). Generate a weight matrix using this pattern, then add noise to the pattern. Afterward, use the weight matrix to perform recall on the noisy pattern, trying to restore it to the original pattern.

### 1.Original pattern image

![截屏2023-05-08 18.34.02](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2018.34.02.png)



### 2.Code：

#### C

[project3.c](./project3.c)

```c
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

```



#### Python

[project3.py](./project3.py):

```python
import json
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# Load patterns and noisy patterns

noisy_level = 15 # 0, 10, 15, 20, 25, 40, 50

noisy_file = "./pattern_"+str(noisy_level)+".json"

with open("./pattern.json", "r") as f:
    original_patterns = json.load(f)

with open(noisy_file, "r") as f:
    noisy_patterns = json.load(f)

n_neuron = 120
n_pattern = 4

# Convert patterns to numpy arrays
for key in original_patterns:
    original_patterns[key] = np.array(original_patterns[key])

for key in noisy_patterns:
    noisy_patterns[key] = np.array(noisy_patterns[key])

# Initialize weight matrix
w = np.zeros((n_neuron, n_neuron))


# Store patterns in the weight matrix
for key in original_patterns:
    pattern = original_patterns[key]
    w += np.outer(pattern, pattern)
w /= n_pattern
np.fill_diagonal(w, 0)
# print(w)


# for key in original_patterns:
#     pattern = original_patterns[key]
#     print(key,pattern)
#     for i in range(n_neuron):
#         for j in range(n_neuron):           
#             w[i, j] += pattern[i] * pattern[j] 
#         # w[i, i] = 0

# w /= n_pattern
# np.fill_diagonal(w, 0)
# # print(w)


# Recall patterns and store in pattern_store
pattern_store = {}

synchronous_update = 1
asynchronous_update = 2
recall_method = 2

for key in noisy_patterns:
    noisy_pattern = noisy_patterns[key]
    original_pattern = original_patterns[key]
    # print(original_pattern)
    # print(noisy_pattern)
    recalled_patterns = [original_pattern.tolist(), noisy_pattern.tolist()]

    while True:
        
        if recall_method == synchronous_update:
            new_pattern = np.where(w @ noisy_pattern >= 0, 1, -1)
            
            if np.array_equal(new_pattern, noisy_pattern):
                break
            
            noisy_pattern = new_pattern
        
        if recall_method == asynchronous_update:

            prev_pattern = noisy_pattern.copy()

            for i in range(n_neuron):
                net_input = w[i] @ noisy_pattern

                noisy_pattern[i] = 1 if net_input >= 0 else -1

            if np.array_equal(noisy_pattern, prev_pattern):
                break
    
        recalled_patterns.append(noisy_pattern.tolist())

    pattern_store[key] = recalled_patterns


def num_to_color(num):
    if num == 1:
        return "white"
    elif num == -1:
        return "red"


max_length = 0
for key, values in pattern_store.items():
    max_length = max(max_length, len(values))

fig, axs = plt.subplots(4, max_length, figsize=(20, 10),gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
spacing = 0.2
row = 0
for key, values in pattern_store.items():
   
    count = 0
    for p in values:
        for i in range(10):
            for j in range(12):
                color = num_to_color(p[i * 12 + j])
                axs[row, count].scatter(j * spacing, -i * spacing, c=color, s=50)
        axs[row, count].axis("off")
        axs[row, count].set_xlim(-0.5 * spacing, 11.5 * spacing)
        axs[row, count].set_ylim(-9.5 * spacing, 0.5 * spacing)
        if count == 0:
            axs[row, count].set_title(f"{key}-original")
        if count == 1:
            axs[row, count].set_title(f"noisy_level:{noisy_level}%")
        if count > 1:
            axs[row, count].set_title(f"{count-1}th-iter")
        count += 1
    row += 1

for i in range(row):
    for j in range(max_length):
         axs[i, j].axis("off")

plt.show()
        

```



## 5.Results

### C

[Original image](./outpat3.txt)：

```
Pattern[0]:
     * * * * * * * *    
     * * * * * * * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * * * * * * * *    
     * * * * * * * *    



Pattern[1]:
     * * * * * * * *    
     * * * * * * * *    
                 * *    
                 * *    
     * * * * * * * *    
     * * * * * * * *    
     * *                
     * * * * * * * *    
     * * * * * * * *    
                        



Pattern[2]:
           * *          
         * * *          
       * * * *          
     * *   * *          
   * *     * *          
   * * * * * * * * *    
   * * * * * * * * *    
           * *          
           * *          
           * *          



Pattern[3]:
                        
   * * * * * * * * *    
   * * * * * * * * *    
   * *                  
   * * * * * * * * *    
   * * * * * * * * *    
   * *           * *    
   * *           * *    
   * * * * * * * * *    
   * * * * * * * * *    


```



The code written in C language is better at restoring the pattern after adding noise. Next is the effect of [recall 15%-level-noise](./outsta3.txt )

```
0-th iteration:
     * * * * * * * *    
     * * *   * * * * *  
     * *         * *    
     *           * *    
       *         * *    
     * *   *     * *    
                 * *    
 *   * *         * *    
     * * * * * * * * *  
   * * *   * * * * *    



1-th iteration:
     * * * * * * * *    
     * * * * * * * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * * * * * * * *    
     * * * * * * * *    



2-th iteration:
     * * * * * * * *    
     * * * * * * * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * *         * *    
     * * * * * * * *    
     * * * * * * * *    



0-th iteration:
 *     *   * * * * *    
     *   * * * * * *   *
 *               * *   *
                 * *    
     * * * * *   * * *  
   * * * * * * * * *    
     * *   * *          
     * *     * *   *    
     * * * * * * * *    
                        



1-th iteration:
     * * * * * * * *    
     * * * * * * * *    
                 * *    
                 * *    
     * *   * *   * *    
     * * * * * * * *    
     * *                
     * * * * * * * *    
     * * * * * * * *    
                        



2-th iteration:
     * * * * * * * *    
     * * * * * * * *    
                 * *    
       *         * *    
     * *   * *   * *    
     * * * * * * * *    
     * *                
     * * * * * * * *    
     * * * * * * * *    
                        



3-th iteration:
     * * * * * * * *    
     * * * * * * * *    
                 * *    
       *         * *    
     * *   * *   * *    
     * * * * * * * *    
     * *                
     * * * * * * * *    
     * * * * * * * *    
                        



0-th iteration:
           * * *        
     *   * * *          
   *   * * * *   *      
     * * * * *   *      
 * *   *                
 * *   * * * * * * *    
     * * * * * *   *    
           * *     *    
         * * *       *  
     *       *   *      



1-th iteration:
           * *          
         * * *          
       * * * *          
     * *   * *          
   * *     * *          
   * * * * * * * * *    
   * * * * * * * * *    
           * *          
           * *          
           * *          



2-th iteration:
           * *          
         * * *          
       * * * *          
     * *   * *          
   * *     * *          
   * * * * * * * * *    
   * * * * * * * * *    
           * *          
           * *          
           * *          



0-th iteration:
             *         *
   *   * *   * * *      
   * * * *   *   * *    
   * *   *     * *     *
   * *     *   * *      
   * * * * * * *        
   * *           * *    
   * *           * *    
   * * * * * * * * *    
   * * * * * * * * *   *



1-th iteration:
                        
   * * * * * * * * *    
   * * * * * * * * *    
   * *                  
   * * * * * * * * *    
   * * * * * * * * *    
   * *           * *    
   * *           * *    
   * * * * * * * * *    
   * * * * * * * * *    



2-th iteration:
                        
   * * * * * * * * *    
   * * * * * * * * *    
   * *                  
   * * * * * * * * *    
   * * * * * * * * *    
   * *           * *    
   * *           * *    
   * * * * * * * * *    
   * * * * * * * * *    




```





### Python

Next, test the memory capacity of the Hopfield network at noise levels of 10%, 15%, 20%, and 25%, respectively.

#### 1.10%-noisy

![截屏2023-05-08 18.40.35](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2018.40.35.png)

#### 2.15%-noisy

![截屏2023-05-08 18.41.22](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2018.41.22.png)

#### 3.20%-noisy

![截屏2023-05-08 18.42.01](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2018.42.01.png)

#### 4.25%-noisy

![截屏2023-05-08 18.42.52](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2018.42.52.png)

It can be seen that the Hopfield network can restore the pattern with very few iterations, demonstrating good memory capacity.

However, the Hopfield network did not successfully recall the original patterns for patterns 0 and 2. A possible reason is that the network is trapped in a locally stable state and cannot escape.

When increasing the noise level to 50%, it can be found that the memory capacity of the Hopfield network is also limited.

![截屏2023-05-08 19.32.56](./Report_project3_EN.assets/%E6%88%AA%E5%B1%8F2023-05-08%2019.32.56.png)
