import java.util.Iterator;
import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;

public class Subset {
   public static void main(String[] args) {
	   int k = Integer.parseInt(args[0]);
	   String seq;
	   RandomizedQueue<String> rq = new RandomizedQueue<String>();
	   while (!StdIn.isEmpty()) {
		   seq = StdIn.readString();
		   rq.enqueue(seq);
	   }
	   Iterator<String> itr = rq.iterator();
	   for (int i = 0; i < k; i++) {
		   StdOut.println(itr.next());
	   }
   }
}