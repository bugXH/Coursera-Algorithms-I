import edu.princeton.cs.algs4.WeightedQuickUnionUF;

/**
 * Programming assignment 1 of Coursera Algorithm I
 * @author Hao
 *
 */
public class Percolation {
    private boolean[] grid = null;
    private int N; // length of grid
    private WeightedQuickUnionUF perc, percExtra; // keep track of the "to-percolate" allocation
    // for surrounding checking
    private int[] surrX = {0, 0, -1, 1}; // up down left right
    private int[] surrY = {-1, 1, 0, 0}; // up down left right
    public Percolation(int N) { // create N-by-N grid, with all sites blocked
        if (N <= 0) {
            throw new java.lang.IllegalArgumentException();
        }
        grid = new boolean[N * N];
        this.N = N;
        // initialize as N * N + 2 sites
        // normal sites are indexed from 0 to N * N - 1
        // 2 extra sites are used as virtual roots for top and bottom,
        // indexed as N * N and N * N + 1 
        perc = new WeightedQuickUnionUF(N * N + 2);
        // the perc_extra is to prevent the "backwash" problem
        // only blocks directly connected to the top are full
        percExtra = new WeightedQuickUnionUF(N * N + 1);
    }
    
    
    public void open(int i, int j) { // open site (row i, column j) if it is not open already
        // check index validity
        if (i < 1 || i > N || j < 1 || j > N) {
            throw new java.lang.IndexOutOfBoundsException();
        }
        // convert to actual index
        int ni = i - 1, nj = j - 1;
        int site_index = ni * N + nj;
        // if not open yet
        if (!grid[site_index]) {
            grid[site_index] = true;
            
            // check surrounding sites
            int surr_i, surr_j;
            for (int k = 0; k < 4; k++) {
                surr_i = ni + surrX[k];
                surr_j = nj + surrY[k];
                // if surrounding site exist and is open already
                // connect to the open part
                int surr_site = surr_i * N + surr_j;
                if (surr_i >= 0 && surr_i < N && surr_j >= 0 && surr_j < N && grid[surr_site]) {
                    perc.union(site_index, surr_site);
                    percExtra.union(site_index, surr_site);
                }
            }
            
            // check top and bottom virtual roots
            if (i == 1) { // connect the sites in the 1st row with the top
                perc.union(site_index, N * N);
                percExtra.union(site_index, N * N);
            }
            if (i == N) { // connect the sites in the last row with the bottom
                perc.union(site_index, N * N + 1);
            }
        }
     }
    
    
    public boolean isOpen(int i, int j) { // is site (row i, column j) open?
        // check index validity
        if (i < 1 || i > N || j < 1 || j > N) {
            throw new java.lang.IndexOutOfBoundsException();
        }
        // convert to actual index
        int ni = i - 1, nj = j - 1;
        return grid[ni * N + nj];
    }
    
    
    public boolean isFull(int i, int j) {  // is site (row i, column j) full?
        return isOpen(i, j) && percExtra.connected((i - 1) * N + (j - 1), N * N);
    }
    
    
    public boolean percolates() {  // does the system percolate?
        return perc.connected(N * N, N * N + 1);
    }
    
    public static void main(String[] args) {
        Percolation p = new Percolation(0);
        p.open(1, 1);
        System.out.println("does the system percolates? " + p.percolates());
        p.open(1, 2);
        System.out.println("does the system percolates? " + p.percolates());
        p.open(2, 2);
        System.out.println("does the system percolates? " + p.percolates());
        p.open(3, 2);
        System.out.println("does the system percolates? " + p.percolates());
        p.open(4, 2);
        System.out.println("does the system percolates? " + p.percolates());
        
    }
}