import java.util.Arrays;
import java.util.ArrayList;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

public class BruteCollinearPoints {
	private LineSegment[] segs;
	
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
		ArrayList<LineSegment> collinear = new ArrayList<LineSegment>();
		
		// sort the points first, all points are sorted in ascending order
		Arrays.sort(points);
		
		// find collinear points
		for (int p = 0; p < points.length - 3; p++) {
			for (int q = p + 1; q < points.length - 2; q++) {
				for (int r = q + 1; r < points.length - 1; r++) {
					for (int s = r + 1; s < points.length; s++) {
						double slope_pq = points[p].slopeTo(points[q]);
						double slope_pr = points[p].slopeTo(points[r]);
						double slope_ps = points[p].slopeTo(points[s]);
						if (slope_pq == slope_pr && slope_pq == slope_ps) {
							collinear.add(new LineSegment(points[p], points[s]));
						}
					}
				}
			}
		}
		segs = (LineSegment[]) collinear.toArray(new LineSegment[collinear.size()]);
	}
	
	public int numberOfSegments() {   // the number of line segments
		return segs.length;
	}
	
	public LineSegment[] segments() {   // the line segments
		return segs;
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