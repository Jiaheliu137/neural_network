import numpy as np

n_neuron = 120
n_pattern = 4
n_row = 10
n_column = 12
noise_rate = 0.20

pattern = np.array([
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
])

w = np.zeros((n_neuron, n_neuron))
v = np.zeros(n_neuron)

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
            print('* ' if v[i*n_column+j] == -1 else '  ', end='')
        print("\n")
    print("\n\n\n")
    input()

def store_pattern():
    global w
    for i in range(n_neuron):
        for j in range(n_neuron):
            w[i][j] = sum(pattern[k][i]*pattern[k][j] for k in range(n_pattern)) / n_pattern
        w[i][i] = 0

def recall_pattern(m):
    global v
    for i in range(n_neuron):
        if np.random.rand() < noise_rate:
            v[i] = -1 if pattern[m][i] == 1 else 1
        else:
            v[i] = pattern[m][i]
    output_state(0)

    k = 1
    while True:
        n_update = 0
        for i in range(n_neuron):
            net = 0
            for j in range(n_neuron):
                net += w[i][j]*v[j]
            vnew = 1 if net >= 0 else -1
            if vnew != v[i]:
                n_update += 1
                v[i] = vnew
        output_state(k)
        k += 1
        if n_update == 0:
            break

def initialization():
    global w
    w = np.zeros((n_neuron, n_neuron))

if __name__ == "__main__":
    for k in range(n_pattern):
        output_pattern(k)

    initialization()
    store_pattern()
    for k in range(n_pattern):
        recall_pattern(k)