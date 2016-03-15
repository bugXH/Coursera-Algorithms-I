import java.util.ArrayList;
import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdOut;

public class Board {
    private int[][] board;
    private int N;
    public Board(int[][] blocks) {  // construct a board from an N-by-N array of blocks
                                    // (where blocks[i][j] = block in row i, column j)
        N = blocks.length;
        board = new int[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = blocks[i][j];
            }
        }
        
        
    }
    
    public int dimension() {  // board dimension N
        return N;
    }
    
    public int hamming() {  // number of blocks out of place
        int n = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // blank is not a block
                if (board[i][j] != (j + 1 + i * N) && board[i][j] != 0) {
                    n++;
                }
            }
        }
        return n;
    }
    
    public int manhattan() {  // sum of Manhattan distances between blocks and goal
        int n = 0;
        int real_i, real_j;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // blank is not a block
                if (board[i][j] != 0) {
                    real_i = (board[i][j] - 1) / N;
                    real_j = (board[i][j] - 1) % N;
                    n += (Math.abs(real_i - i) + Math.abs(real_j - j));
                }
            }
        }
        return n;
    }
    
    public boolean isGoal() {   // is this board the goal board?
        return hamming() == 0;
    }
    
    public Board twin() {   // a board that is obtained by exchanging any pair of blocks
        int i = 0, j = 0;
        int[][] twinboard = copynewboard();
        
        // blank cannot be exchanged
        out: for (i = 0; i < N; i++) {
            for (j = 0; j < N - 1; j++) {
                if (board[i][j] != 0 && board[i][j + 1] != 0) {
                    break out;
                }
            }
        }
        // exchange the given one and the one to its right
        int tmp = twinboard[i][j];
        twinboard[i][j] = twinboard[i][j + 1];
        twinboard[i][j + 1] = tmp;
        return new Board(twinboard);
        
    }
    
    public boolean equals(Object y) {      // does this board equal y?
        // check reference
        if (this == y) {
            return true;
        }
        
        // check class
        if (y == null || this.getClass() != y.getClass()) {
            return false;
        }
        
        // check dimension
        Board that = (Board) y;
        if (this.N != that.N) {
            return false;
        }
        
        // check length
        for (int i = 0; i < N; i++) {
            if (this.board[i].length != that.board[i].length) {
                return false;
            }
            
            // check each element
            for (int j = 0; j < N; j++) {
                if (this.board[i][j] != that.board[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    public Iterable<Board> neighbors() {   // all neighboring boards (those that can be reached in one move)
        ArrayList<Board> neighbor = new ArrayList<Board>();
        // get the indices of the blank block
        int i = 0, j = 0;
        out: for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (board[i][j] == 0) {
                    break out;
                }
            }
        }
        
        // 4 directions for one move in total: up down left right
        // up
        if (i != N - 1) {
            int[][] newboard = copynewboard();
            // exchange
            int tmp = newboard[i + 1][j];
            newboard[i + 1][j] = newboard[i][j];
            newboard[i][j] = tmp;
            neighbor.add(new Board(newboard));
        }
        
        //down
        if (i != 0) {
            int[][] newboard = copynewboard();
            // exchange
            int tmp = newboard[i - 1][j];
            newboard[i - 1][j] = newboard[i][j];
            newboard[i][j] = tmp;
            neighbor.add(new Board(newboard));
        }
        
        //left
        if (j != N - 1) {
            int[][] newboard = copynewboard();
            // exchange
            int tmp = newboard[i][j + 1];
            newboard[i][j + 1] = newboard[i][j];
            newboard[i][j] = tmp;
            neighbor.add(new Board(newboard));
        }
        
        //right
        if (j != 0) {
            int[][] newboard = copynewboard();
            // exchange
            int tmp = newboard[i][j - 1];
            newboard[i][j - 1] = newboard[i][j];
            newboard[i][j] = tmp;
            neighbor.add(new Board(newboard));
        }
        return neighbor;
    }
    
    private int[][] copynewboard() {
        int[][] newboard = new int[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                newboard[i][j] = board[i][j];
            }
        }
        return newboard;
    }
    
    public String toString() { // string representation of this board (in the output format specified below)
        StringBuilder str = new StringBuilder();
        str.append(Integer.toString(N) + "\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (j < N - 1) {
                    str.append(Integer.toString(board[i][j]).concat("  "));
                }
                else {
                    str.append(Integer.toString(board[i][j]).concat("\n"));
                }
            }
        }
        return str.toString();
    }

    public static void main(String[] args) { // unit tests (not graded)
        // create initial board from file
        In in = new In(args[0]);
        int N = in.readInt();
        int[][] blocks = new int[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                blocks[i][j] = in.readInt();
        
        Board initial = new Board(blocks);
        StdOut.println("hamming " + initial.hamming());
        StdOut.println("manhattan " + initial.manhattan());
        StdOut.println(initial.toString());
        
        blocks[0][0] = 99;
        StdOut.println("hamming " + initial.hamming());
        StdOut.println("manhattan " + initial.manhattan());
        StdOut.println(initial.toString());
        
        
        Board twin = initial.twin();
        StdOut.println(twin.toString());
        
        StdOut.println(initial.toString());
    }
}