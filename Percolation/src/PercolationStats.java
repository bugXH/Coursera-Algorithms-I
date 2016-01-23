import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;

public class PercolationStats {
    private double[] thresh = null;
    private double mean, stddev, conflow, confhigh;
    public PercolationStats(int N, int T)  {  // perform T independent experiments on an N-by-N grid
        if (N <= 0 || T <= 0) {
            throw new java.lang.IllegalArgumentException();
        }
        thresh = new double[T];
        for (int t = 0; t < T; t++) {
            int count = 0;
            Percolation p = new Percolation(N);
            int ran_limit = N * N;
            int i, j, site_index;
            do {
                site_index = StdRandom.uniform(ran_limit);
                i = site_index / N + 1;
                j = site_index % N + 1;
                while (p.isOpen(i, j)) {
                    site_index = StdRandom.uniform(ran_limit);
                    i = site_index / N + 1;
                    j = site_index % N + 1;
                }
                p.open(i, j);
                count++;
            } while(!p.percolates());
            thresh[t] = (double) count / ran_limit;
            
            // calculations
            mean = StdStats.mean(thresh);
            stddev = StdStats.stddev(thresh);
            conflow = mean - 1.96 * stddev / Math.sqrt(T);
            confhigh = mean + 1.96 * stddev / Math.sqrt(T);
        }
    }
    
    public double mean() {  // sample mean of percolation threshold
        return mean;
    }
    
    public double stddev() {    // sample standard deviation of percolation threshold
        return stddev;
    }
    
    public double confidenceLo() {  // low  endpoint of 95% confidence interval
        return conflow;
    }
    
    public double confidenceHi() { // high endpoint of 95% confidence interval
        return confhigh;
    }
    
    public static void main(String[] args) {
        int N, T;
        N = Integer.parseInt(args[0]);
        T = Integer.parseInt(args[1]);
        PercolationStats pstats = new PercolationStats(N, T);
        System.out.println("mean                    = " + pstats.mean());
        System.out.println("stddev                  = " + pstats.stddev());
        System.out.println("95% confidence interval = " + pstats.confidenceLo() + ", " + pstats.confidenceHi());
    }

}
