import matplotlib.pyplot as plt



pattern = """
 C * * * C * * C * C C C * C C 
* * * C * B * C * * * * * * * 
 * * * B * * * * * * * * C * * 
C * * B * * * C * * C C * C C 
 * B * * * * B * * * C * * C C 
* B * B B B * B * * * * * C C 
 B B B B * * B * * C * C C * * 
B * * B B B * * * * C * * * C 
 B B * B * B B B * * * * * C C 
* * * B * * B * B * B * * * * 
 * * * * * * * * * * * B * * B 
A A A A A * * * * * * * * * B 
 * A * A * * A A * * * * B B B 
A * * A A A A * A A * * * * B 
 A A A A A A A * A A * * * B * 
"""

colors = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
    '*': 'white',
}

rows = [list(row) for row in pattern.strip().split('\n')]
rows = [[char for char in row if char != ' '] for row in rows]  # 空白文字を削除
height = len(rows)
width = max(len(row) for row in rows)

plt.figure(figsize=(width/2, height/2))

for y in range(height):
    for x in range(len(rows[y])):
        char = rows[y][x]
        color = colors[char]
        plt.scatter(x, height-y-1, c=color, alpha=0.5, s=300)

plt.axis('off')

plt.show()