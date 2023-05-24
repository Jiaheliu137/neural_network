This repository is used to record the project for the AIZU University course: Neural Networks I: Fundamental Theory and Applications.

Run the following code to use this repository.

```python
git clone https://github.com/Jiaheliu137/neural_network
cd neural_network
pip install -r requirements.txt
```

1. project2(4-bit parity check):[Report_project2_CN.md](./project2/Report_project2_CN.md),  [Report_project2_EN.md](./project2/Report_project2_EN.md), [Report_project2_CN.pdf](./project2/Report_project2_CN.pdf),   [Report_project2_EN.pdf](./project2/Report_project2_EN.pdf)

   ```python
   cd project2
   ```

   

   1. Draw a graph

      ```python
      python project2.py
      ```

   2. Draw a graph matrix

      ```python
      python draw_plot.py
      ```

      

1. project3(Hopfield Network):[Report_project3_CN.md](./project3/Report_project3_CN.md),  [Report_project3_EN.md](./project3/Report_project3_EN.md), [Report_project3_CN.pdf](./project3/Report_project3_CN.pdf),   [Report_project3_EN.pdf](./project3/Report_project3_EN.pdf)

   ```python
   cd project3
   ```

   

   - Visualizing images of raw data patterns.
     ```python
     python draw_pattern.py
     ```

   - Adding noise to raw data patterns.
     ```python
     python noise.py -nl 0.15
     ```

     - $nl \in [0, 1]$

   - Recall noised pattern and visulize

     ```python
     python project3.py -nl 15
     ```

     - $nl \in [0, 1]$ and $nl$ shuld be integer

2. project4(WTA network, Iris dataset):[Report_project4_CN.md](./project4/Report_project4_CN.md),  [Report_project4_EN.md](./project4/Report_project4_EN.md), [Report_project4_CN.pdf](./project4/Report_project4_CN.pdf),   [Report_project4_EN.pdf](./project4/Report_project4_EN.pdf)

   ```python
   cd project4
   python project4.py -v 3 -c 6
   ```

   - `-v` |` --visualize : The dimensionality of visualization`, $v \in \{2, 3\}$
   - -c | --cluster : Number of clusters, $c \geq 1$

4. project5(SOFM, Iris dataset):[Report_project5_CN.md](./project5/Report_project5_CN.md),  [Report_project5_EN.md](./project5/Report_project5_EN.md), [Report_project5_CN.pdf](./project5/Report_project5_CN.pdf),   [Report_project5_EN.pdf](./project5/Report_project5_EN.pdf)

   - Use WTA to generate the cluster file
     ```python
     python output_cluster.py [-c <num>]
     ```

   - Use SOFM to compressing data dimensions and visualize				
     ```python
     python project5.py [-c <num>]
     ```

The SOFM (Self-Organizing Feature Map) algorithm has multiple nested loops and requires plotting, which makes its execution speed relatively slow

**eg.**

```python
cd project5
python output_cluster.py -c 3
python project5.py -c 3
```

- -c | --cluster : Number of clusters,the value is recommended $2 \leq c \leq 6$









​					
