import java.util.Arrays;
import java.util.ArrayList;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;


public class FastCollinearPoints {
	private LineSegment[] segs;
	
    public FastCollinearPoints(Point[] points) { // finds all line segments containing 4 or more points
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
		// store the collinear line segment
		ArrayList<LineSegment> collinear = new ArrayList<LineSegment>();
		ArrayList<String> collinearString = new ArrayList<String>();
		
		// a copy of sorted points
		Point[] pointsorted = Arrays.copyOf(points, points.length);
		for (Point pointp: points) {
			// sort according to slope order first
			Arrays.sort(pointsorted, pointp.slopeOrder());
			
			// a temp list for storing the current collinear points
			ArrayList<Point> colltmp = new ArrayList<Point>();
			
			// the slope to compare with
			double slope = pointp.slopeTo(pointsorted[0]);
			
			//init the start and end points as the first point of the slope
			Point start = pointsorted[0], end = pointsorted[0];
			
			// compare the slope one by one
			for (Point pointq: pointsorted) {
				if (pointq == pointp) {
					continue;
				}
				double slope1 = pointp.slopeTo(pointq);
				if (slope1 == slope) {
					colltmp.add(pointq);
					// get the new start and end points, if necessary
					if (pointq.compareTo(start) < 0) {
						start = pointq;
					}
					if (pointq.compareTo(end) > 0) {
						end = pointq;
					}
				}
				// if new slope occur, check whether there are more than 3 points already
				else {
					if (colltmp.size() >= 3) {
						// compare the start and end points with point p
						if (pointp.compareTo(start) < 0) {
							start = pointp;
						}
						if (pointp.compareTo(end) > 0) {
							end = pointp;
						}
						LineSegment newseg = new LineSegment(start, end);
						String newsegString = newseg.toString();
						if (!collinearString.contains(newsegString)) {
							collinearString.add(newsegString);
							collinear.add(newseg);
						}
						
					}
					// clean the temp list, add the new point
					colltmp.clear();
					colltmp.add(pointq);
					
					// set new start end points, set new slope to compare with
					start = pointq;
					end = pointq;
					slope = slope1;
				}
			}
		}
		segs = (LineSegment[]) collinear.toArray(new LineSegment[collinear.size()]);
    }
   
    public int numberOfSegments() {   // the number of line segments
	   return segs.length;
    }
   
    public LineSegment[] segments() {      // the line segments
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
	    FastCollinearPoints collinear = new FastCollinearPoints(points);
	    for (LineSegment segment : collinear.segments()) {
	        StdOut.println(segment);
	        segment.draw();
	    }
	}
}