import random

n_neuron = 120
n_pattern = 4
n_row = 10
n_column = 12
noise_rate = 0.20

pattern = [
  [1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,   
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,  
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,  
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
  [1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
   1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,
   1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
   1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1]
]


w = [[0]*n_neuron for _ in range(n_neuron)]
v = [0]*n_neuron

def output_pattern(k):
    print("Pattern[{}]:\n".format(k))
    for i in range(n_row):
        for j in range(n_column):
            print("* " if pattern[k][i*n_column+j]==-1 else "  ", end="")
        print("\n")
    print("\n\n\n")
    input()

def output_state(k):
    print("{}-th iteration:\n".format(k))
    for i in range(n_row):
        for j in range(n_column):
            print("* " if v[i*n_column+j]==-1 else "  ", end="")
        print("\n")
    print("\n\n\n")
    input()

def store_pattern():
    for i in range(n_neuron):
        for j in range(n_neuron):
            w[i][j] = 0
            for k in range(n_pattern):
                w[i][j] += pattern[k][i]*pattern[k][j]
            w[i][j] /= n_pattern
        w[i][i] = 0

def recall_pattern(m):
    global v
    r = 0
    for i in range(n_neuron):
        r = random.random()
        if r < noise_rate:
            v[i] = -1 if pattern[m][i] == 1 else 1
        else:
            v[i] = pattern[m][i]
    output_state(0)

    k = 1
    while True:
        n_update = 0
        v_new = [0]*n_neuron
        for i in range(n_neuron):
            net = 0
            for j in range(n_neuron):
                net += w[i][j]*v[j]
            v_new[i] = 1 if net >= 0 else -1
            if v_new[i] != v[i]:
                n_update += 1
        v = v_new
        output_state(k)
        k += 1
        if n_update == 0:
            break

def initialization():
    global w
    w = [[0]*n_neuron for _ in range(n_neuron)]

def main():
    for k in range(n_pattern):
        output_pattern(k)

    initialization()
    store_pattern()
    for k in range(n_pattern):
        recall_pattern(k)

main()
