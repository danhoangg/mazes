# C++ Maze generator and solver

My ASCII maze generator using the ASCII box-drawing characters. It involves 3 maze solving algorithms, DFS, BFS and greedy best-first search. 

https://github.com/user-attachments/assets/1d65b829-d5df-46a8-839e-a85c0b168d17

## Running it ▶️
Must be run in linux \
`./main`

Or compile it yourself \
`g++ main.cpp -o main` \
`./main`

To make the maze bigger, zoom out in your terminal. It creates the maze by getting window dimensions.

## How does it work ⚙️

### Maze generation 🏗️

1. The maze begins completely closed off. Both algorithms pick edges to open a route between one square to another.

2. 
    1. Initially used a disjoint-set and union find to check if the start and end squares were connected. Then it would randomly add edges until the start and end were under the same parent.
    2. However this left some squares unconnected to the main maze, so I switched to using Prim's algorithm that used a priority queue with a random custom comparator to ensure each maze was generated randomly.
3. During generation, an adjacency map is constructed to be used in the maze solving algorithms

### Maze solving 🧩

I used 3 different algorithms

#### DFS
Standard DFS algorithm, I coloured in the current path red and all visited squares as blue. 

#### BFS
Standard BFS algorithm, I coloured in all the squares that were seen as blue. 
It kept a map of predecessors for each square it visited; this was used in the backtracking function which coloured the path found red by going backwards through the predecessors

#### A*
Not-so-standard A* but rather a greedy best-first search as it relies solely on the heuristic function.
I am prioritising speed over optimality in this case, the heuristic I use is the the manhattan distance doubled, this allows the algorithm to path aggressively towards the target square.
This heuristic is neither admissible nor consistent and therefore sacrifices optimality.


