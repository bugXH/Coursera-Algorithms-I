import java.util.Iterator;
import edu.princeton.cs.algs4.StdOut;

public class Deque<Item> implements Iterable<Item> {
	
	private class Node {
		public Item value;
		public Node prev, next;
		
		public Node() {
			value = null;
			prev = null;
			next = null;
		}
	}
	
	private int size;
	private Node head, tail;
	
	public Deque() {
		size = 0;
		head = tail = null;
	}
	
	public boolean isEmpty() {  // is the deque empty?
		return (size == 0);
	}
	
	public int size() {  // return the number of items on the deque
		return size;
	}
	
	public void addFirst(Item item) {  // add the item to the front
		if (item == null) { // cannot add null item
			throw new java.lang.NullPointerException();
		}
		// add a node to the front
		Node toadd = new Node();
		toadd.value = item;
		toadd.next = head;
		if (head != null) {
			head.prev = toadd;
		}
		// make the new node the new head
		head = toadd;
		size++;
		// if this is the 1st node added, make it the tail as well
		if (tail == null) {
			tail = head;
		}
	}
	
	public void addLast(Item item) {  // add the item to the end
		if (item == null) {
			throw new java.lang.NullPointerException();
		}
		Node toadd = new Node();
		toadd.value = item;
		toadd.prev = tail;
		if (tail != null) {
			tail.next = toadd;
		}
		tail = toadd;
		size++;
		if (head == null) {
			head = tail;
		}
	}
	
	public Item removeFirst() { // remove and return the item from the front
		if (size == 0) {
			throw new java.util.NoSuchElementException();
		}
		Item item = head.value;
		if (size == 1) {
			head = tail = null;
		}
		else {
			head = head.next;
			head.prev = null;
		}
		size--;
		return item;
	}
	
	public Item removeLast() { // remove and return the item from the end
		if (size == 0) {
			throw new java.util.NoSuchElementException();
		}
		Item item = tail.value;
		if (size == 1) {
			head = tail = null;
		}
		else {
			tail = tail.prev;
			tail.next = null;
		}
		size--;
		return item;
	}
		
	public Iterator<Item> iterator() { // return an iterator over items in order from front to end
		return new DequeIterator();
	}
	
	private class DequeIterator implements Iterator<Item> {
		private Node current = head;
		
		@Override
		public boolean hasNext() {
			return current != null;
		}
		
		@Override
		public Item next() {
			if (!hasNext()) {
				throw new java.util.NoSuchElementException();
			}
			Item item = current.value;
			current = current.next;
			return item;
		}
		
		@Override
		public void remove() {
			throw new java.lang.UnsupportedOperationException();
		}
	}
	
	public static void main(String[] args) {
		Deque<Integer> dq = new Deque<Integer>();
		StdOut.println("size "  + dq.size());
		dq.addFirst(3);
		dq.addLast(4);
		StdOut.println("size "  + dq.size());
		Iterator<Integer> itr;
		for (itr = dq.iterator(); itr.hasNext(); ) {
			StdOut.println(itr.next());
		}
		itr = dq.iterator();
		StdOut.println("empty? " + dq.isEmpty());
	}
}
