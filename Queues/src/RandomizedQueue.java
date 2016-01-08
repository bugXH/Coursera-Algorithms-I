import java.util.Iterator;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdOut;

public class RandomizedQueue<Item> implements Iterable<Item> {
	
	private Item[] queue;
	private int size;
	private int defaultsize;
	
	public RandomizedQueue() { // construct an empty randomized queue
		defaultsize = 10;
		size = 0;
		queue = (Item[])new Object[defaultsize];
	}
	
	public boolean isEmpty() {  // is the queue empty?
		return (size == 0);
	}
	
	public int size() {  // return the number of items on the queue
		return size;
	}
	
	
	public void enqueue(Item item) { // add the item
		if (item == null) {
			throw new java.lang.NullPointerException();
		}
		// check if the current queue needs to be extended
		if (size == queue.length) {
			// create a new queue that is of twice the length
			// and copy all the elements
			Item[] newqueue = (Item[]) new Object[size * 2];
			for (int i = 0; i < size; i++) {
				newqueue[i] = queue[i];
			}
			queue = newqueue;
		}
		queue[size] = item;
		size++;
	}
	
	public Item dequeue() { // remove and return a random item
		if (isEmpty()) {
			throw new java.util.NoSuchElementException();
		}
		// get a random index
		int ranidx = getranidx();
		Item item = queue[ranidx];
		size--;
		
		// since the iterator is over items in random order
		// just copy the last item to the dequeued position
		// no need to move items up
		queue[ranidx] = queue[size];
		queue[size] = null;
		
		// for memory usage's sake, shrink the queue if necessary
		if (queue.length > defaultsize * 4 && size <= queue.length / 4) {
			Item[] newqueue = (Item[]) new Object[queue.length / 2];
			for (int i = 0; i < size; i++) {
				newqueue[i] = queue[i];
			}
			queue = newqueue;
		}
		
		return item;
	}
	
	public Item sample() {  // return (but do not remove) a random item
		if (isEmpty()) {
			throw new java.util.NoSuchElementException();
		}
		return queue[getranidx()];
	}
	
	private int getranidx() {
		while (true) {
			int ranidx = StdRandom.uniform(size);
			if (queue[ranidx] != null) {
				return ranidx;
			}
		}
	}
	
	public Iterator<Item> iterator() { // return an independent iterator over items in random order
		return new RanQIterator(queue, size);
	}
	
	private class RanQIterator implements Iterator<Item> {
		private Item[] iterqueue;
		private int iteridx;
		
		public RanQIterator(Item[] queue, int size) {
			iterqueue = (Item[]) new Object[size];
			iteridx = 0;
			// copy all the data
			for (int i = 0; i < size; i++) {
				iterqueue[i] = queue[i];
			}
			
			// randomly shuffle the iterqueue
			for (int i = 0; i < size; i++) {
				int swapidx = StdRandom.uniform(i + 1);
				Item tmp = iterqueue[i];
				iterqueue[i] = iterqueue[swapidx];
				iterqueue[swapidx] = tmp;
			}
		}
		
		@Override
		public boolean hasNext() {
			return (iteridx < iterqueue.length);
		}
		
		@Override
		public Item next() {
			if (!hasNext()) {
				throw new java.util.NoSuchElementException();
			}
			Item item = iterqueue[iteridx++];
			return item;
		}
		
		@Override
		public void remove() {
			throw new java.lang.UnsupportedOperationException();
		}
	}
	
	public static void main(String[] args) {
		RandomizedQueue<Double> rq = new RandomizedQueue<Double>();
		StdOut.println("Empty? " + rq.isEmpty());
		rq.enqueue(0.1);
		rq.enqueue(1.5);
		rq.enqueue(2.3);
		rq.enqueue(3.3);
		StdOut.println("Empty? " + rq.isEmpty());
		Iterator<Double> itr;
		for (itr = rq.iterator(); itr.hasNext(); ) {
			StdOut.println(itr.next());
		}
		rq.dequeue();
		rq.dequeue();
		rq.dequeue();
		rq.dequeue();
		StdOut.println("Empty? " + rq.isEmpty());
	}
}
