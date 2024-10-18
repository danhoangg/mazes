#include <bits/stdc++.h>
//#include <windows.h>
#include <sys/ioctl.h>
#include <unistd.h>
using namespace std;

/**
 * Standard disjoint union find set using rank.
 * Used to efficiently check if a solution exists between start and end in maze.
 */
struct UnionFindSet {
    vector<int> parent, rank;

    UnionFindSet(int n) : parent(n), rank(n, 1) {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void merge(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    bool together(int x, int y) {
        return find(x) == find(y);
    }
};

vector<pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
int delay;  // delay to view the algorithms visually

/**
 * Gets a random edge in the maze
 * @param n rows of maze
 * @param m cols of maze
 * @return random edge
 */
pair<int, int> randomEdge(int n, int m) {
    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<> dist_r(0, n - 1);
    uniform_int_distribution<> dist_c(0, m - 1);
    int r = dist_r(gen);
    int c = dist_c(gen);

    vector<pair<int, int>> adj;
    for (const pair<int, int>& d : directions) {
        int dr = r + d.first, dc = c + d.second;
        if (dr >= 0 && dr < n && dc >= 0 && dc < m) {
            adj.push_back({dr, dc});
        }
    }

    uniform_int_distribution<> dist_choice(0, adj.size() - 1);
    int choice = dist_choice(gen);

    return {r * m + c, adj[choice].first * m + adj[choice].second};
}

/**
 * Prints the maze to screen
 * @param visual the visual grid to print
 */
void showMaze(vector<vector<string>>& visual) {
    for (auto i : visual) {
        for (auto j : i) {
            cout << j;
        }
        cout << endl;
    }
}

/**
 * Creates the initial grid of boxes for the maze
 * @param n rows of maze
 * @param m cols of maze
 * @return 2d vector of strings which make up the maze
 */
vector<vector<string>> createVisualGrid(int n, int m) {
    vector<vector<string>> visual(2 * n + 1, vector<string>(2 * m + 1));
    visual[0][0] = "┏";
    for (int i = 1; i <= 2 * m - 1; i++) {
        if (i % 2 == 0) {
            visual[0][i] = "┳";
        } else {
            visual[0][i] = "━";
        }
    }
    visual[0][2 * m] = "┓";

    for (int i = 1; i <= 2 * n - 1; i += 2) {
        visual[i][0] = "┃";
        for (int j = 1; j <= 2 * m - 1; j++) {
            if (j % 2 == 0) {
                visual[i][j] = "┃";
            } else {
                visual[i][j] = " ";
            }
        }
        visual[i][2 * m] = "┃";

        visual[i+1][0] = "┣";
        for (int j = 1; j <= 2 * m - 1; j++) {
            if (j % 2 == 0) {
                visual[i+1][j] = "╋";
            } else {
                visual[i+1][j] = "━";
            }
        }
        visual[i+1][2 * m] = "┫";
    }

    visual[2 * n][0] = "┗";
    for (int i = 1; i <= 2 * m - 1; i++) {
        if (i % 2 == 0) {
            visual[2 * n][i] = "┻";
        } else {
            visual[2 * n][i] = "━";
        }
    }
    visual[2 * n][2 * m] = "┛";

    visual[1][0] = " ";

    return visual;
}

/**
 * When opening a way between 2 boxes the corners need to be changed
 * to make the ascii maze look nice
 * @param c the current ascii box character at that position
 * @param d the direction, 0 for top, subsequent values are counted clockwise
 * @return The box character that it should be changed to
 */
string switchBoxLine(string c, int d) {
    if (d == 0) {
        if (c == "╋") return "┻";
        if (c == "┳" || c == "┏" || c == "┓") return "━";
        if (c == "┫") return "┛";
        if (c == "┣") return "┗";
    } else if (d == 1) {
        if (c == "╋") return "┣";
        if (c == "┫" || c == "┓" || c == "┛") return "┃";
        if (c == "┳") return "┏";
        if (c == "┻") return "┗";
    } else if (d == 2) {
        if (c == "╋") return "┳";
        if (c == "┫") return "┓";
        if (c == "┻" || c == "┛" || c == "┗") return "━";
        if (c == "┣") return "┏";
    } else if (d == 3) {
        if (c == "╋") return "┫";
        if (c == "┻") return "┛";
        if (c == "┣" || c == "┏" || c == "┗") return "┃";
        if (c == "┳") return "┓";
    }

    return c;
}

/**
 * Used for opening the maze up on the visual grid
 * @param edge edge being opened
 * @param n rows of the maze
 * @param m cols of the maze
 * @param US the union find set
 * @param visual the visual grid
 */
void changeVisualGrid(pair<int, int> edge, int n, int m, vector<vector<string>>& visual) {
    int r = edge.first / m, c = edge.first % m, dr = edge.second / m, dc = edge.second % m;
    int x = r + dr + 1, y = c + dc + 1;
    visual[x][y] = " ";

    if (r == dr) {
        visual[x - 1][y] = switchBoxLine(visual[x - 1][y] , 0);
        visual[x + 1][y] = switchBoxLine(visual[x + 1][y], 2);
    } else {
        visual[x][y - 1] = switchBoxLine(visual[x][y - 1], 3);
        visual[x][y + 1] = switchBoxLine(visual[x][y + 1], 1);
    }
}

/**
 * For visualising the search algorithms using coloured blocks
 * @param row row to change character of
 * @param col col to change character of
 */
void moveCursorTo(int row, int col) {
    cout << "\033[" << row << ";" << col << "H";
}

/**
 * Returns twice the manhattan distance between two points
 * @param r row of first point
 * @param c col of first point
 * @param n row of second point
 * @param m col of second point
 * @return twice the manhattan distance between two points
 */
int manhattan(int r, int c, int n, int m) {
    return 2 * (abs(r - n) + abs(c - m));
}

/**
 * Backtracks the path that was found by BFS or A*
 * @param p hashmap of predecessors
 * @param start starting square
 * @param target target square
 * @param n rows of maze
 * @param m cols of maze
 */
void backtrackPath(unordered_map<int, int>& p, int start, int target, int n, int m) {
    int cur = target;
    int prev = target;

    int backtrackDelay = delay == 0 ? 0 : max(1, delay / 4);

    while ((cur = p[cur]) != start) {
        this_thread::sleep_for(std::chrono::milliseconds(backtrackDelay));
        int x = cur / m, y = cur % m;
        int dx = prev / m, dy = prev % m;

        moveCursorTo(2 * x + 2, 2 * y + 2);
        cout << "\033[31m█\033[0m";
        cout.flush();
        moveCursorTo(x + dx + 2, y + dy + 2);
        cout << "\033[31m█\033[0m";
        cout.flush();

        prev = cur;
    }
}

/**
 * The A* algorithm, purely heuristic so technically a greedy best-first search.
 * The heuristic used is not admissible or consistent as speed is being prioritised over optimality
 * @param start starting square
 * @param target target square
 * @param n rows of the maze
 * @param m cols of the maze
 * @param adj adjacency map
 * @return true if there exists a path, false otherwise
 */
bool astar(int start, int target, int n, int m, unordered_map<int, vector<int>>& adj) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    unordered_set<int> visited;
    unordered_map<int, int> predecessor;

    int tx = target / m, ty = target % m;

    pq.push({manhattan(start / m, start % m, tx, ty), start});
    visited.insert(start);

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (u == target) {
            break;
        }

        int x = u / m, y = u % m;
        moveCursorTo(2 * x + 2, 2 * y + 2);
        cout << "\033[34m█\033[0m";
        cout.flush();

        for (const int v : adj[u]) {
            if (visited.find(v) == visited.end()) {
                this_thread::sleep_for(chrono::milliseconds(delay));
                visited.insert(v);

                int dx = v / m, dy = v % m;
                moveCursorTo(x + dx + 2, y + dy + 2);
                cout << "\033[34m█\033[0m";
                cout.flush();

                pq.push({manhattan(dx, dy, tx, ty), v});

                predecessor[v] = u;
            }
        }
    }

    if (predecessor.find(target) != predecessor.end()) {
        backtrackPath(predecessor, start, target, n, m);
        return true;
    }

    return false;
}

/**
 * A BFS algorithm
 * @param start starting square
 * @param target target square
 * @param n rows of maze
 * @param m cols of maze
 * @param adj adjacency map
 * @return true if there exists a path, false otherwise
 */
bool bfs(int start, int target, int n, int m, unordered_map<int, vector<int>>& adj) {
    queue<int> q;
    unordered_set<int> visited;
    unordered_map<int, int> predecessor;

    int tx = target / m, ty = target % m;

    q.push(start);
    visited.insert(start);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        if (u == target) {
            break;
        }

        int x = u / m, y = u % m;
        moveCursorTo(2 * x + 2, 2 * y + 2);
        cout << "\033[34m█\033[0m";
        cout.flush();

        for (const int v : adj[u]) {
            if (visited.find(v) == visited.end()) {
                this_thread::sleep_for(chrono::milliseconds(delay));
                visited.insert(v);

                int dx = v / m, dy = v % m;
                moveCursorTo(x + dx + 2, y + dy + 2);
                cout << "\033[34m█\033[0m";
                cout.flush();

                q.push(v);

                predecessor[v] = u;
            }
        }
    }

    if (predecessor.find(target) != predecessor.end()) {
        backtrackPath(predecessor, start, target, n, m);
        return true;
    }

    return false;
}

/**
 * A DFS algorithm
 * @param u the current node
 * @param target target square
 * @param n rows of maze
 * @param m cols of maze
 * @param adj adjacency map
 * @param visited hashset of visited nodes
 * @return true if there exists a path, false otherwise
 */
bool dfs(int u, int target, int n, int m, unordered_map<int, vector<int>>& adj, vector<bool>& visited) {
    this_thread::sleep_for(chrono::milliseconds(delay));
    if (u == target) return true;
    visited[u] = true;
    int r = u / m, c = u % m;

    moveCursorTo(2 * r + 2, 2 * c + 2);
    cout << "\033[31m█\033[0m";
    cout.flush();

    for (const int v : adj[u]) {
        if (!visited[v]) {
            int dr = v / m, dc = v % m;
            moveCursorTo(r + dr + 2, c + dc + 2);
            cout << "\033[31m█\033[0m";
            cout.flush();
            if (dfs(v, target, n, m, adj, visited)) return true;
            moveCursorTo(r + dr + 2, c + dc + 2);
            cout << "\033[34m█\033[0m";
            cout.flush();
        }
    }

    moveCursorTo(2 * r + 2, 2 * c + 2);
    cout << "\033[34m█\033[0m";
    cout.flush();

    return false;
}

/**
 * Generates a random target square on the board
 * @param n rows of the maze
 * @param m cols of the maze
 * @return A random square on the board
 */
int randTarget(int n, int m) {
    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution dist(1, n * m - 1);

    return dist(gen);
}

/**
 * Generates all the edges in a maze
 * @param n rows of the maze
 * @param m cols of the maze
 * @return All the edges in a maze
 */
vector<pair<int, int>> generateEdges(int n, int m) {
    vector<pair<int, int>> edges;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < m; c++) {
            int cur = r * m + c;

            if (c < m - 1) {
                edges.push_back({cur, cur + 1});
            }

            if (r < n - 1) {
                edges.push_back({cur, cur + m});
            }
        }
    }

    return edges;
}

/**
 * Struct used for randomising the priority queue in prims
 */
struct RandCMP {
    bool operator()(const pair<int, int>& a, const pair<int, int>& b) const {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, 1);

        return dist(gen);
    }
};

/**
 * Prims algorithm to connect all the nodes in the maze together.
 * Uses priority queue with random comparator to randomise maze
 * @param n rows of the maze
 * @param m cols of the maze
 * @param visual the visual grid
 * @param adj adjacency map
 */
void prims(int n, int m, vector<vector<string>>& visual, unordered_map<int, vector<int>>& adj) {
    vector<int> nmDirections = {1, -1, m, -m};

    unordered_set<int> mst;
    priority_queue<pair<int, int>, vector<pair<int, int>>, RandCMP> pq;
    pq.push({1, 0});
    pq.push({m, 0});

    while (mst.size() < n * m && !pq.empty()) {
        int cur = pq.top().first;
        int pred = pq.top().second;
        pq.pop();
        if (mst.find(cur) == mst.end()) {
            mst.insert(cur);

            adj[cur].push_back(pred);
            adj[pred].push_back(cur);

            changeVisualGrid({cur, pred}, n, m, visual);

            for (const int d : nmDirections) {
                int nei = cur + d;
                if (d == 1 && (cur + 1) % m == 0) continue;
                if (d == -1 && cur % m == 0) continue;
                if (nei >= 0 && nei < n * m) {
                    pq.push({nei, cur});
                }
            }
        }
    }
}

int main() {
    //SetConsoleOutputCP(CP_UTF8);

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    // row and col for
    int n = (w.ws_row - 1) / 2 - 2, m = (w.ws_col - 1) / 2;
    //int n = 200, m = 400;
    int _delay = 0;
    /*cout << "Enter n: ";
    cin >> n;
    cout << "Enter m: ";
    cin >> m;*/
    cout << "Enter delay (ms): ";
    cin >> _delay;
    delay = _delay;

    vector<vector<string>> visual = createVisualGrid(n, m);
    unordered_map<int, vector<int>> adj;

    int target = randTarget(n, m);

    /*
     * Old code for generating the maze
     * I think prims algorithm is more intuitive than this however much slower generation
    UnionFindSet US = UnionFindSet(n * m);
    while (!US.together(0, target)) {
        pair<int, int> edge = randomEdge(n, m);
        if (US.together(edge.first, edge.second)) continue;
        US.merge(edge.first, edge.second);
        changeVisualGrid(edge, n, m, US, visual);

        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }
    */

    prims(n, m, visual, adj);

    visual[(target / m) * 2 + 1][(target % m) * 2 + 1] = "X";
    showMaze(visual);

    // pq to show which was fastest algorithm at the end
    priority_queue<pair<chrono::duration<double>, string>, vector<pair<chrono::duration<double>, string>>, greater<>> durations;

    cout << "Press enter to solve...";
    cin.ignore();
    cin.get();
    system("clear");
    showMaze(visual);

    vector<bool> visited(n * m, false);

    auto start = chrono::high_resolution_clock::now();
    dfs(0, target, n, m, adj, visited);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    durations.push({duration, "DFS"});

    moveCursorTo(2*n + 2, 0);
    cout << "Solved in " << duration.count() << " seconds" << endl;

    cout << "Press enter to solve...";
    cin.get();
    system("clear");
    moveCursorTo(0, 0);
    showMaze(visual);

    start = chrono::high_resolution_clock::now();
    bfs(0, target, n, m, adj);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    durations.push({duration, "BFS"});

    moveCursorTo(2*n + 2, 0);
    cout << "Solved in " << duration.count() << " seconds" << endl;

    cout << "Press enter to solve...";
    cin.get();
    system("clear");
    moveCursorTo(0, 0);
    showMaze(visual);

    start = chrono::high_resolution_clock::now();
    astar(0, target, n, m, adj);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    durations.push({duration, "A*"});

    moveCursorTo(2*n + 2, 0);
    cout << "Solved in " << duration.count() << " seconds" << endl;

    cout << durations.top().second << " wins!" << endl;

    return 0;
}