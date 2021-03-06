import java.util.Arrays;
import java.util.ArrayList;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

public class BruteCollinearPoints {
    private ArrayList<LineSegment> collinear;
    
    public BruteCollinearPoints(Point[] points) { // finds all line segments containing 4 points
        // check null arguments
        if (points == null) {
            throw new java.lang.NullPointerException("Array is null!");
        }
        for (int i = 0; i < points.length; i++) {
            if (points[i] == null) {
                throw new java.lang.NullPointerException("Null entry in array!");
            }
        }
        // check duplications
        for (int i = 0; i < points.length - 1; i++) {
            for (int j = i + 1; j < points.length; j++) {
                if (points[i].compareTo(points[j]) == 0) {
                    throw new java.lang.IllegalArgumentException("A repeated point is in the array!");
                }
            }
        }
        collinear = new ArrayList<LineSegment>();
        
        // sort the points first, all points are sorted in ascending order
        Point[] pointscopy = Arrays.copyOf(points, points.length);
        Arrays.sort(pointscopy);
        
        // find collinear points
        for (int p = 0; p < pointscopy.length - 3; p++) {
            for (int q = p + 1; q < pointscopy.length - 2; q++) {
                for (int r = q + 1; r < pointscopy.length - 1; r++) {
                    for (int s = r + 1; s < pointscopy.length; s++) {
                        double slope_pq = pointscopy[p].slopeTo(pointscopy[q]);
                        double slope_pr = pointscopy[p].slopeTo(pointscopy[r]);
                        double slope_ps = pointscopy[p].slopeTo(pointscopy[s]);
                        if (slope_pq == slope_pr && slope_pq == slope_ps) {
                            collinear.add(new LineSegment(pointscopy[p], pointscopy[s]));
                        }
                    }
                }
            }
        }
    }
    
    public int numberOfSegments() {   // the number of line segments
        return collinear.size();
    }
    
    public LineSegment[] segments() {   // the line segments
        return (LineSegment[]) collinear.toArray(new LineSegment[collinear.size()]);
    }
    
    public static void main(String[] args) {
        // read the N points from a file
        In in = new In(args[0]);
        int N = in.readInt();
        Point[] points = new Point[N];
        for (int i = 0; i < N; i++) {
            int x = in.readInt();
            int y = in.readInt();
            points[i] = new Point(x, y);
        }

        // draw the points
        StdDraw.show(0);
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        for (Point p : points) {
            p.draw();
        }
        StdDraw.show();

        // print and draw the line segments
        BruteCollinearPoints collinear = new BruteCollinearPoints(points);
        for (LineSegment segment : collinear.segments()) {
            StdOut.println(segment);
            segment.draw();
        }
    }
}
