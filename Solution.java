package leetcode;

import org.omg.CORBA.MARSHAL;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.net.ServerSocket;
import java.util.*;
import java.util.concurrent.*;

public class Solution {
    
    
    
    //    public int minDepth(TreeNode root) {
//        if (root == null) return 0;
//        LinkedList<TreeNode> queue = new LinkedList<>();
//        queue.add(root);
//        int level = 1;
//        int len;
//        while ((len = queue.size()) != 0) {
//            for (int i = 0; i < len; i++) {
//                TreeNode node = queue.removeFirst();
//                if (node.left == null && node.right == null) return level;
//                if (node.left != null) queue.add(node.left);
//                if (node.right != null) queue.add(node.right);
//            }
//            level++;
//        }
//        return level;
//
//    }

//    public int numSquares(int n) {
//        ArrayList<Integer> squares = new ArrayList<>();
//        int[] dp = new int[n + 1];
//        dp[0] = 0;
//        dp[1] = 1;
//        for (int i = 1; i * i <= n; i++) {
//            squares.add(i * i);
//        }
//        for (int i = 2; i <= n; i++) {
//            for (int square : squares) {
//                if (square > i) break;
//                else {
//                    if (dp[i] == 0) {
//                        dp[i] = dp[i - square] + 1;
//                    } else {
//                        dp[i] = Math.min(dp[i - square] + 1, dp[i]);
//                    }
//                }
//            }
//        }
//        return dp[n];
//
//    }

//    public int numSquares(int n) {
//        ArrayList<Integer> squares = new ArrayList<>();
//        for (int i = 0; i * i <= n; i++) {
//            squares.add(i * i);
//        }
//        HashSet<Integer> queue = new HashSet<>();
//        queue.add(n);
//        int level = 1;
//        while (queue.size() > 0) {
//            HashSet<Integer> newQueue = new HashSet<>();
//            for (Integer candidate : queue) {
//                for (Integer square : squares) {
//                    if (candidate.equals(square)) return level;
//                    if (square > candidate) break;
//                    newQueue.add(candidate - square);
//                }
//            }
//            level++;
//            queue = newQueue;
//        }
//
//        return level;
//    }

//    public static void main(String[] args) {
//        String[] test = {"hot", "dot", "dog", "lot", "log", "cog"};
//        System.out.println(new NewSolution().ladderLength("hit", "cog", Arrays.asList(test)));
//    }
//
//    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
//        List<List<Integer>> adjacent = new ArrayList<>();
//        HashMap<String, Integer> map = new HashMap<>();
//        int count = 0;
//
//        adjacent.add(new ArrayList<>());
//        map.put(beginWord, count);
//        count++;
//        StringBuilder beginSb = new StringBuilder(beginWord);
//        for (int i = 0; i < beginWord.length(); i++) {
//            char originC = beginSb.charAt(i);
//            beginSb.setCharAt(i, '*');
//            adjacent.add(new ArrayList<>());
//            map.put(beginSb.toString(), count);
//            count++;
//            adjacent.get(map.get(beginWord)).add(map.get(beginSb.toString()));
//            adjacent.get(map.get(beginSb.toString())).add(map.get(beginWord));
//            beginSb.setCharAt(i, originC);
//        }
//
//        for (String word : wordList) {
//            adjacent.add(new ArrayList<>());
//            map.put(word, count);
//            count++;
//            StringBuilder sb = new StringBuilder(word);
//            for (int i = 0; i < sb.length(); i++) {
//                char originC = sb.charAt(i);
//                sb.setCharAt(i, '*');
//                if (!map.containsKey(sb.toString())) {
//                    adjacent.add(new ArrayList<>());
//                    map.put(sb.toString(), count);
//                    count++;
//                }
//                adjacent.get(map.get(word)).add(map.get(sb.toString()));
//                adjacent.get(map.get(sb.toString())).add(map.get(word));
//                sb.setCharAt(i, originC);
//            }
//        }
//        if (!map.containsKey(endWord)) return 0;
//        HashSet<Integer> visit = new HashSet<>();
//        LinkedList<Integer> queue = new LinkedList<>();
//        queue.add(map.get(beginWord));
//        int target = map.get(endWord);
//        int depth = 0;
//        int len;
//
//        while ((len = queue.size()) > 0) {
//            for (int i = 0; i < len; i++) {
//                int index = queue.removeFirst();
//                visit.add(index);
//                if (index == target) return depth / 2 + 1;
//                for (int j = 0; j < adjacent.get(index).size(); j++) {
//                    if (!visit.contains(adjacent.get(index).get(j))) {
//                        queue.add(adjacent.get(index).get(j));
//                    }
//                }
//            }
//            depth++;
//        }
//        return 0;
//    }

//    public static void main(String[] args) {
//        int[][] test = {{1, 2, 7}, {3, 6, 7}};
//
//        System.out.println(new NewSolution().numBusesToDestination(test, 1, 6));
//    }
//
//
//    public int numBusesToDestination(int[][] routes, int source, int target) {
//        if (source == target) return 0;
//        List<List<Integer>> adjacent = new ArrayList<>();
//        HashSet<Integer> visit = new HashSet<>();
//        HashSet<Integer> targetBus = new HashSet<>();
//        for (int[] route : routes) {
//            adjacent.add(new ArrayList<>());
//            Arrays.sort(route);
//        }
//
//        for (int i = 0; i < routes.length; i++) {
//            for (int j = i + 1; j < routes.length; j++) {
//                if (intersect(routes[i], routes[j])) {
//                    adjacent.get(i).add(j);
//                    adjacent.get(j).add(i);
//                }
//            }
//        }
//
//        LinkedList<Integer> queue = new LinkedList<>();
//
//        for (int i = 0; i < routes.length; i++) {
//            int[] route = routes[i];
//            if (Arrays.binarySearch(route, target) >= 0) {
//                targetBus.add(i);
//            }
//            if (Arrays.binarySearch(route, source) >= 0) {
//                visit.add(i);
//                queue.add(i);
//            }
//        }
//
//        int num = 1;
//        int len;
//        while ((len = queue.size()) != 0) {
//            for (int i = 0; i < len; i++) {
//                int route = queue.removeFirst();
//                if (targetBus.contains(route)) {
//                    return num;
//                }
//                for (int j = 0; j < adjacent.get(route).size(); j++) {
//                    if (!visit.contains(adjacent.get(route).get(j))) {
//                        visit.add(adjacent.get(route).get(j));
//                        queue.add(adjacent.get(route).get(j));
//                    }
//                }
//            }
//            num++;
//        }
//
//        return -1;
//    }
//
//    public boolean intersect(int[] a, int[] b) {
//        int i = 0;
//        int j = 0;
//        while (i < a.length && j < b.length) {
//            if (a[i] == b[j]) return true;
//            else if (a[i] > b[j]) j++;
//            else i++;
//        }
//        return false;
//    }


//    public int minimumEffortPath(int[][] heights) {
//        List<int[]> edges = new ArrayList<>();
//        for (int i = 0; i < heights.length; i++) {
//            for (int j = 0; j < heights[0].length; j++) {
//                int point = i * heights[0].length + j;
//                if (i != 0) {
//                    int[] edge = new int[3];
//                    edge[0] = point;
//                    edge[1] = (i - 1) * heights[0].length + j;
//                    edge[2] = Math.abs(heights[i][j] - heights[i - 1][j]);
//                    edges.add(edge);
//                }
//                if (j != 0) {
//                    int[] edge = new int[3];
//                    edge[0] = point;
//                    edge[1] = i * heights[0].length + j - 1;
//                    edge[2] = Math.abs(heights[i][j] - heights[i][j - 1]);
//                    edges.add(edge);
//                }
//            }
//        }
//
//        Collections.sort(edges, new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[2] - o2[2];
//            }
//        });
//
//        UnionFind u = new UnionFind(heights.length * heights[0].length);
//        int result = 0;
//        for (int[] edge : edges) {
//            u.union(edge[0], edge[1]);
//            if (u.find(0) == u.find(heights.length * heights[0].length - 1)) {
//                result = edge[2];
//                break;
//            }
//        }
//        return result;
//
//    }
//
//    class UnionFind {
//        int[] parent;
//        int[] size;
//
//        UnionFind(int cap) {
//            parent = new int[cap];
//            size = new int[cap];
//            for (int i = 0; i < cap; i++) {
//                parent[i] = i;
//                size[i] = 1;
//            }
//        }
//
//        int find(int index) {
//            if (parent[index] != index) {
//                return parent[index] = find(parent[index]);
//            }
//            return index;
//        }
//
//        void union(int left, int right) {
//            int leftParent = find(left);
//            int rightParent = find(right);
//            if (size[leftParent] > size[rightParent]) {
//                int tmp = leftParent;
//                leftParent = rightParent;
//                rightParent = tmp;
//            }
//            parent[leftParent] = parent[rightParent];
//            size[rightParent] += size[leftParent];
//        }
//    }


//    public int longestConsecutive(int[] nums) {
//        HashSet<Integer> set = new HashSet<>();
//        int max = 0;
//        for (int num : nums) {
//            set.add(num);
//        }
//
//
//        for (int num : nums) {
//            if (!set.contains(num - 1)) {
//                int len = 1;
//                int next = num + 1;
//                while (set.contains(next)) {
//                    len++;
//                    next++;
//                }
//                max = Math.max(max, len);
//            }
//        }
//        return max;
//    }


//    public int[] findOrder(int numCourses, int[][] prerequisites) {
//        int[] result = new int[numCourses];
//        int[] empty = {};
//        int[] inDegree = new int[numCourses];
//        List<List<Integer>> adjacent = new ArrayList<>();
//        for (int i = 0; i < numCourses; i++) {
//            adjacent.add(new ArrayList<>());
//        }
//
//        for (int[] prerequisite : prerequisites) {
//            inDegree[prerequisite[0]]++;
//            adjacent.get(prerequisite[1]).add(prerequisite[0]);
//        }
//
//        LinkedList<Integer> queue = new LinkedList<>();
//        int count = 0;
//        for (int i = 0; i < inDegree.length; i++) {
//            if (inDegree[i] == 0) queue.addLast(i);
//        }
//
//        while (queue.size() > 0) {
//            int course = queue.poll();
//            result[count] = course;
//            count++;
//            for (int i = 0; i < adjacent.get(course).size(); i++) {
//                int index = adjacent.get(course).get(i);
//                inDegree[index]--;
//                if (inDegree[index] == 0) queue.addLast(index);
//            }
//        }
//
//        return count == numCourses ? result : empty;
//    }
    
//    public int majorityElement(int[] nums) {
//        int candidate = -1;
//        int count = 0;
//        for (int num : nums) {
//            if (count == 0) {
//                candidate = num;
//                count++;
//            } else if (candidate == num) {
//                count++;
//            } else {
//                count--;
//            }
//
//        }
//        return candidate;
//    }

//    public void solve(char[][] board) {
//        for (int i = 0; i < board.length; i++) {
//            if (board[i][0] != 'X' && board[i][0] != '1') visit(board, i, 0);
//            if (board[i][board[0].length - 1] != 'X' && board[i][board[0].length - 1] != '1')
//                visit(board, i, board[0].length - 1);
//        }
//
//        for (int j = 0; j < board[0].length; j++) {
//            if (board[0][j] != 'X' && board[0][j] != '1') visit(board, 0, j);
//            if (board[board.length - 1][j] != 'X' && board[board.length - 1][j] != '1')
//                visit(board, board.length - 1, j);
//        }
//
//        for (int i = 0; i < board.length; i++) {
//            for (int j = 0; j < board[0].length; j++) {
//                if (board[i][j] == '1') {
//                    board[i][j] = 'O';
//                } else {
//                    board[i][j] = 'X';
//                }
//            }
//        }
//    }
//
//    public void visit(char[][] board, int i, int j) {
//        if (board[i][j] == 'X' || board[i][j] == '1') return;
//        board[i][j] = '1';
//        if (i > 0) {
//            visit(board, i - 1, j);
//        }
//        if (i < board.length - 1) {
//            visit(board, i + 1, j);
//        }
//        if (j > 0) {
//            visit(board, i, j - 1);
//        }
//        if (j < board[0].length - 1) {
//            visit(board, i, j + 1);
//        }
//    }

//    leetcode 99
//    public static void main(String[] args) {
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(3);
//        root.left.right = new TreeNode(2);
//        new NewSolution().recoverTree(root);
//    }
//
//    public void recoverTree(TreeNode root) {
//        List<TreeNode> list = new ArrayList<>();
//        inOrder(root, list);
//        int first = -1;
//        int second = -1;
//        for (int i = 0; i < list.size() - 1; i++) {
//            if (list.get(i).val > list.get(i + 1).val) {
//                if (first == -1) {
//                    first = i;
//                } else {
//                    second = i;
//                }
//            }
//        }
//        if (second != -1) {
//            second++;
//            int tmp = list.get(first).val;
//            list.get(first).val = list.get(second).val;
//            list.get(second).val = tmp;
//        } else {
//            int tmp = list.get(first).val;
//            list.get(first).val = list.get(first + 1).val;
//            list.get(first + 1).val = tmp;
//        }
//    }
//
//    public void inOrder(TreeNode root, List<TreeNode> list) {
//        if (root == null) return;
//        inOrder(root.left, list);
//        list.add(root);
//        inOrder(root.right, list);
//    }    
    
    
    
    
//    DST Leetcode 207
//    public boolean canFinish(int numCourses, int[][] prerequisites) {
//        List<Integer>[] courses = new List[numCourses];
//        int[] visited = new int[numCourses];
//        int count = 1;
//        for (int i = 0; i < courses.length; i++) {
//            courses[i] = new ArrayList<Integer>();
//        }
//        for (int[] prerequisite : prerequisites) {
//            courses[prerequisite[0]].add(prerequisite[1]);
//        }
//
//        for (int i = 0; i < courses.length; i++) {
//            if (visited[i] != 0) continue;
//            if (!visit(courses, visited, i)) return false;
//        }
//        return true;
//    }
//
//    public boolean visit(List<Integer>[] courses, int[] visited, int courseNo) {
//        if (visited[courseNo] == 1) return false;
//        if (visited[courseNo] == -1) return true;
//        visited[courseNo] = 1;
//        List<Integer> course = courses[courseNo];
//        for (int i = 0; i < course.size(); i++) {
//            if (!visit(courses, visited, course.get(i))) return false;
//        }
//        visited[courseNo] = -1;
//        return true;
//    }


//    leetcode 547
//    public static void main(String[] args) {
//        int[][] test = {{1,0,0,1},{0,1,1,0},{0,1,1,1},{1,0,1,1}};
//        System.out.println(new NewSolution().findCircleNum(test));
//    }
//
//
//    public int findCircleNum(int[][] isConnected) {
//        int count = 0;
//        HashSet<Integer> connect = new HashSet<>();
//        for (int i = 0; i < isConnected.length; i++) {
//            if (connect.contains(i)) continue;
//            count++;
//            visit(isConnected, connect, i);
//        }
//        return count;
//    }
//
//    public void visit(int[][] isConnected, HashSet<Integer> connect, int city) {
//        if (connect.contains(city)) return;
//        connect.add(city);
//        for (int i = 0; i < isConnected[city].length; i++) {
//            if (i == city) continue;
//            if (isConnected[city][i] == 1) {
//                visit(isConnected, connect, i);
//            }
//        }
//    }

//    leetcode 220
//    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
//        HashMap<Long, Long> buckets = new HashMap<>();
//        long w = (long) (t + 1);
//        for (int i = 0; i < nums.length; i++) {
//            long id = getId(nums[i], w);
//            if (buckets.containsKey(id)) return true;
//            if (buckets.containsKey(id + 1) && buckets.get(id + 1) - nums[i] <= t) return true;
//            if (buckets.containsKey(id - 1) && nums[i] - buckets.get(id - 1) <= t) return true;
//            buckets.put(id, (long)nums[i]);
//            if (i >= k) buckets.remove(getId(nums[i - k], w));
//        }
//        return false;
//    }
//
//    public long getId(long num, long w) {
//        if (num >= 0) return num / w;
//        return (num + 1) / w - 1;
//    }


//    leetcode 219
//    public boolean containsNearbyDuplicate(int[] nums, int k) {
//        HashMap<Integer, Integer> map = new HashMap<>();
//        for (int i = 0; i < nums.length; i++) {
//            if (map.containsKey(nums[i])) {
//                int index = map.get(nums[i]);
//                if (i - index <= k) return true;
//            }
//            map.put(nums[i], i);
//        }
//        return false;
//    }


//    leetcode 217
//    public boolean containsDuplicate(int[] nums) {
//        HashSet<Integer> set = new HashSet<>();
//        for (int num : nums) {
//            if (!set.add(num)) return true;
//        }
//        return false;
//    }

//    leetcode 721
//    public List<List<String>> accountsMerge(List<List<String>> accounts) {
//        HashMap<String, String> emailName = new HashMap<>();
//        HashMap<String, Integer> emailIndex = new HashMap<>();
//        int count = 0;
//        for (int i = 0; i < accounts.size(); i++) {
//            List<String> account = accounts.get(i);
//            String name = account.get(0);
//            for (int j = 1; j < account.size(); j++) {
//                String email = account.get(j);
//                emailName.put(email, name);
//                emailIndex.put(email, count);
//                count++;
//            }
//        }
//        UnionFind unionfind = new UnionFind(count);
//        for (int i = 0; i < accounts.size(); i++) {
//            List<String> account = accounts.get(i);
//            int index = emailIndex.get(account.get(1));
//            for (int j = 2; j < account.size(); j++) {
//                String email = account.get(j);
//                int index2 = emailIndex.get(email);
//                unionfind.union(index, index2);
//            }
//        }
//
//        HashMap<Integer, List<String>> collect = new HashMap<>();
//        for (String email : emailIndex.keySet()) {
//            int index = emailIndex.get(email);
//            int parent = unionfind.find(index);
//            if (collect.containsKey(parent)) {
//                List list = collect.get(parent);
//                list.add(email);
//            } else {
//                List<String> list = new ArrayList<>();
//                list.add(email);
//                collect.put(parent, list);
//            }
//        }
//        ArrayList<List<String>> result = new ArrayList<>();
//        for (Integer key : collect.keySet()) {
//            List<String> emails = collect.get(key);
//            String name = emailName.get(emails.get(0));
//            Collections.sort(emails);
//            ArrayList<String> account = new ArrayList<>();
//            account.add(name);
//            account.addAll(emails);
//            result.add(account);
//        }
//        return result;
//    }
//
//    class UnionFind {
//        int[] parent;
//
//        UnionFind(int cap) {
//            parent = new int[cap];
//            for (int i = 0; i < cap; i++) {
//                parent[i] = i;
//            }
//        }
//
//        int find(int index) {
//            if (parent[index] == index) return index;
//            parent[index] = find(parent[index]);
//            return parent[index];
//        }
//
//        void union(int left, int right) {
//            int parentLeft = find(left);
//            int parentRight = find(right);
//            parent[parentLeft] = parentRight;
//        }
//    }


//    leetcode 679
//    boolean result = false;
//    static final double min = 1e-6;
//    int count = 0;
//
//    public boolean judgePoint24(int[] nums) {
//        ArrayList<Double> list = new ArrayList<>();
//        for (int num : nums) {
//            list.add((double) num);
//        }
//        judge(list);
//        return result;
//    }
//
//    public void judge(ArrayList<Double> list) {
//        count++;
//        if (list.size() == 1) {
//            if (Math.abs(list.get(0) - 24) < min) result = true;
//            return;
//        }
//
//        for (int i = 0; i < list.size(); i++) {
//            for (int j = 1; i + j < list.size(); j++) {
//                ArrayList<Double> newList = new ArrayList<>();
//                for (int k = 0; k < list.size(); k++) {
//                    if (k != i && k != (i + j)) {
//                        newList.add(list.get(k));
//                    }
//                }
//                double n1 = list.get(i);
//                double n2 = list.get(i + j);
//
//                newList.add(n1 + n2);
//                judge(newList);
//                newList.remove(newList.size() - 1);
//
//                newList.add(n1 * n2);
//                judge((newList));
//                newList.remove(newList.size() - 1);
//
//                newList.add(n1 - n2);
//                judge((newList));
//                newList.remove(newList.size() - 1);
//                newList.add(n2 - n1);
//                judge((newList));
//                newList.remove(newList.size() - 1);
//
//                newList.add(n1 / n2);
//                judge((newList));
//                newList.remove(newList.size() - 1);
//                newList.add(n2 / n1);
//                judge((newList));
//                newList.remove(newList.size() - 1);
//            }
//        }
//
//    }


//    public boolean isPalindrome(ListNode head) {
//        if (head.next == null) return true;
//        ListNode p = head;
//        ListNode q = head;
//        while (q != null) {
//            p = p.next;
//            q = q.next;
//            if (q == null) break;
//            q = q.next;
//        }
//
//        ListNode mid = p;
//        ListNode a = p;
//        ListNode b = p.next;
//        while (b != null) {
//            ListNode c = b.next;
//            b.next = a;
//            a = b;
//            b = c;
//        }
//        mid.next = null;
//
//        while (a != null) {
//            if (a.val != head.val) return false;
//            a = a.next;
//            head = head.next;
//        }
//        return true;
//    }
    
//    leetcode 146    
//    HashMap<Integer, CacheNode> cache = new HashMap<>();
//    CacheNode head = null;
//    CacheNode tail = null;
//    int cap;
//    int size;
//
//
//    public LRUCache(int capacity) {
//        cap = capacity;
//        size = 0;
//    }
//
//    public int get(int key) {
//        CacheNode n = cache.get(key);
//        if (n != null) {
//            moveToHead(n);
//            return n.val;
//        }
//        return -1;
//    }
//
//    public void put(int key, int value) {
//        if (cache.containsKey(key)) {
//            cache.get(key).val = value;
//            moveToHead(cache.get(key));
//        } else {
//            CacheNode n = new CacheNode(value, key);
//            if (size == cap) {
//                cache.remove(tail.key);
//                if (head == tail) {
//                    head = null;
//                    tail = null;
//                } else {
//                    tail = tail.pre;
//                    tail.next = null;
//                }
//                size--;
//            }
//
//            if (head == null) {
//                head = n;
//                tail = n;
//            } else {
//                head.pre = n;
//                n.next = head;
//                head = n;
//            }
//            size++;
//            cache.put(key, n);
//        }
//    }
//
//    public void moveToHead(CacheNode node) {
//        if (size == 1 || head == node) return;
//        if (tail == node) {
//            tail = node.pre;
//            tail.next = null;
//            node.pre = null;
//            node.next = head;
//            head.pre = node;
//            head = node;
//        } else {
//            node.pre.next = node.next;
//            node.next.pre = node.pre;
//            node.pre = null;
//            node.next = head;
//            head.pre = node;
//            head = node;
//        }
//    }
//
//    class CacheNode {
//
//        CacheNode next;
//        CacheNode pre;
//        int val;
//        int key;
//
//        CacheNode(int v, int k) {
//            val = v;
//            key = k;
//        }
//
//    }
    
    
//    leetcode 394
//    public static void main(String[] args) {
//        System.out.println(new NewSolution().decodeString("3[a2[c]]"));
//    }
//
//
//    public String decodeString(String s) {
//        StringBuilder sb = new StringBuilder();
//        int i = 0;
//        while (i < s.length()) {
//            char c = s.charAt(i);
//            if (c >= '0' && c <= '9') {
//                StringBuilder num = new StringBuilder();
//                num.append(c);
//                i++;
//                while (s.charAt(i) >= '0' && s.charAt(i) <= '9') {
//                    num.append(s.charAt(i));
//                    i++;
//                }
//                i++;
//                int number = Integer.parseInt(num.toString());
//                StringBuilder sub = new StringBuilder("");
//                int count = 0;
//                while (!(count == 0 && s.charAt(i) == ']')) {
//                    if (s.charAt(i) == '[') count++;
//                    if (s.charAt(i) == ']') count--;
//                    sub.append(s.charAt(i));
//                    i++;
//                }
//                String subString = decodeString(sub.toString());
//                for (int j = 0; j < number; j++) {
//                    sb.append(subString);
//                }
//                i++;
//                continue;
//            }
//            sb.append(c);
//            i++;
//        }
//        return sb.toString();
//    }


//    leetcode 9
//    public boolean isPalindrome(int x) {
//        if (x < 0 || (x !=0 &&x % 10 == 0)) return false;
//        int reverse = 0;
//        int x1 = x;
//        while (x1 > 0) {
//            reverse = reverse * 10 + x1 % 10;
//            x1 /= 10;
//        }
//        return reverse == x;
//    }
    
//    leetcode124
//    int result = Integer.MIN_VALUE;
//
//    public int maxPathSum(TreeNode root) {
//        dfs(root);
//        return result;
//    }
//
//    public int dfs(TreeNode root) {
//        if (root == null) return 0;
//        int leftMax = Math.max(dfs(root.left),0);
//        int rightMax = Math.max(dfs(root.right),0);
//        int sum = leftMax + rightMax + root.val;
//        result = Math.max(result, sum);
//        return Math.max(leftMax, rightMax) + root.val;
//    }


//    public static void main(String[] args) {
//        char[][] test = {{'1', '1', '1'}, {'0', '1', '0'}, {'1', '0', '0'}, {'1', '0', '1'}};
//        System.out.println(new NewSolution().numIslands(test));
//    }

//    leetcode200
//    public int numIslands(char[][] grid) {
//        int result = 0;
//        for (int i = 0; i < grid.length; i++) {
//            for (int j = 0; j < grid[0].length; j++) {
//
//                if (grid[i][j] == '1') {
//                    result++;
//                }
//                visit(i, j, grid);
//            }
//        }
//        return result;
//    }
//
//    public void visit(int i, int j, char[][] grid) {
//        if (i >= grid.length || i < 0 || j < 0 || j >= grid[0].length) return;
//        if (grid[i][j] == '0') return;
//        grid[i][j] = '0';
//        visit(i + 1, j, grid);
//        visit(i - 1, j, grid);
//        visit(i, j - 1, grid);
//        visit(i, j + 1, grid);
//    }
    
//    leetcode 5
//    public String longestPalindrome(String s) {
//        boolean[][] dp = new boolean[s.length()][s.length()];
//        int maxLen = 1;
//        String ans = s.substring(0, 1);
//        for (int k = 0; k < s.length(); k++) {
//            for (int i = 0; i + k < s.length(); i++) {
//                int j = i + k;
//                if (k == 0) dp[i][i] = true;
//                else if (k == 1 && s.charAt(i) == s.charAt(j)) {
//                    dp[i][j] = true;
//                } else if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
//                    dp[i][j] = true;
//                }
//                if (dp[i][j] && maxLen < k + 1) {
//                    maxLen = k + 1;
//                    ans = s.substring(i, j + 1);
//                }
//            }
//        }
//        return ans;
//    }    
    
    
//    leetcode131
//    List<List<String>> result = new ArrayList<>();
//    ArrayDeque<String> path = new ArrayDeque<>();
//
//    public List<List<String>> partition(String s) {
//        boolean[][] dp = new boolean[s.length()][s.length()];
//        for (int k = 0; k < s.length(); k++) {
//            for (int i = 0; i + k < s.length(); i++) {
//                int j = i + k;
//                if (k == 0) dp[i][i] = true;
//                else if (k == 1 && s.charAt(i) == s.charAt(j)) {
//                    dp[i][j] = true;
//                } else if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
//                    dp[i][j] = true;
//                }
//            }
//        }
//        getPartition(s, dp, 0);
//        return result;
//    }
//
//    public void getPartition(String s, boolean[][] dp, int start) {
//        if (start >= s.length()) {
//            result.add(new ArrayList<>(path));
//            return;
//        }
//        for (int i = 0; start + i < s.length(); i++) {
//            if (dp[start][start + i]) {
//                path.add(s.substring(start, start + i + 1));
//                getPartition(s, dp, start + i + 1);
//                path.removeLast();
//            }
//        }
//    }    
    
    
    
//    leetcode 95    
//    public List<TreeNode> generateTrees(int n) {
//        return buildTree(1, n);
//    }
//
//    public List<TreeNode> buildTree(int left, int right) {
//        List<TreeNode> result = new ArrayList<>();
//        if (left > right) {
//            result.add(null);
//            return result;
//        }
//        for (int i = left; i <= right; i++) {
//            List<TreeNode> leftNode = buildTree(left, i - 1);
//            List<TreeNode> rightNode = buildTree(i + 1, right);
//            for (int i1 = 0; i1 < leftNode.size(); i1++) {
//                for (int i2 = 0; i2 < rightNode.size(); i2++) {
//                    TreeNode node = new TreeNode(i);
//                    node.left = leftNode.get(i1);
//                    node.right = rightNode.get(i2);
//                    result.add(node);
//                }
//            }
//        }
//        return result;
//    }    
    
    
//    leetcode331
//    public static void main(String[] args) {
//        String test = "9,9,91,#,#,9,#,49,#,#,#";
//        System.out.println(new NewSolution().isValidSerialization(test));
//    }
//
//    public boolean isValidSerialization(String preorder) {
//        int need = 1;
//        for (int i = 0; i < preorder.length(); i++) {
//            if (need == 0) return false;
//            if (preorder.charAt(i) == '#') {
//                need--;
//            }
//            if (preorder.charAt(i) >= '0' && preorder.charAt(i) <= '9') {
//                while (i + 1 < preorder.length() && preorder.charAt(i + 1) >= '0' && preorder.charAt(i + 1) <= '9') i++;
//                need++;
//            }
//        }
//        return need == 0;
//    }    
    
//    leetcode 437
//    int result = 0;
//
//    public int pathSum(TreeNode root, int sum) {
//        HashMap<Integer, Integer> map = new HashMap<>();
//        map.put(0, 1);
//        findSum(root, 0, sum, map);
//        return result;
//    }
//
//    public void findSum(TreeNode root, int nowSum, int target, HashMap<Integer, Integer> map) {
//        if (root == null) return;
//        nowSum += root.val;
//        if (map.containsKey(nowSum - target)) {
//            result += map.get(nowSum - target);
//        }
//        if (map.containsKey(nowSum)) {
//            map.put(nowSum, map.get(nowSum) + 1);
//        } else {
//            map.put(nowSum, 1);
//        }
//        findSum(root.left, nowSum, target, map);
//        findSum(root.right, nowSum, target, map);
//        map.put(nowSum, map.get(nowSum) - 1);
//    }    
    
//    leetcode 96    
//    public static void main(String[] args) {
//        System.out.println(new NewSolution().numTrees(3));
//    }
//
//    public int numTrees(int n) {
//        int[] dp = new int[n + 1];
//        dp[1] = 1;
//        for (int i = 2; i <= n; i++) {
//            for (int j = 1; j <= i; j++) {
//                int left = j - 1;
//                int right = i - j;
//                int count = 0;
//                if (left == 0) count = dp[right];
//                else if (right == 0) count = dp[left];
//                else count = dp[left] * dp[right];
//                dp[i] += count;
//            }
//        }
//        return dp[n];
//    }    
    
//    leetcode 129    
//    int result = 0;
//
//    public int sumNumbers(TreeNode root) {
//        if (root == null) return result;
//        getSum(root, 0);
//        return result;
//    }
//
//    public void getSum(TreeNode root, int sum) {
//        sum = sum * 10 + root.val;
//        if (root.left == null && root.right == null) {
//            result += sum;
//            return;
//        }
//        if (root.left != null) getSum(root.left, sum);
//        if (root.right != null) getSum(root.right, sum);
//    }
    

//    leetcode 112    
//    boolean result = false;
//
//    public boolean hasPathSum(TreeNode root, int targetSum) {
//        if (root == null) return false;
//        getSum(root, 0, targetSum);
//        return result;
//    }
//
//    public void getSum(TreeNode root, int sum, int targetSum) {
//        sum += root.val;
//        if (root.left == null && root.right == null) {
//            if (sum == targetSum) result = true;
//            return;
//        }
//
//        if (root.left != null) getSum(root.left, sum, targetSum);
//        if (root.right != null) getSum(root.right, sum, targetSum);
//
//    }
    
    
//    leetcode 101
//    public boolean isSymmetric(TreeNode root) {
//        if (root == null) return true;
//        return judge(root.left, root.right);
//    }
//
//    public boolean judge(TreeNode left, TreeNode right) {
//        if (left == null && right == null) return true;
//        if (left == null || right == null) return false;
//        if (left.val != right.val) return false;
//        return judge(left.right, right.left) && judge(left.left, right.right);
//    }
    
//    Leetcode 257    
//    ArrayList<String> result = new ArrayList<>();
//
//    public List<String> binaryTreePaths(TreeNode root) {
//        if (root == null) return result;
//        LinkedList<Integer> path = new LinkedList<>();
//        addNode(path, root);
//        return result;
//    }
//
//    public void addNode(LinkedList<Integer> path, TreeNode root) {
//        path.addLast(root.val);
//        if (root.left == null && root.right == null) {
//            Iterator<Integer> it = path.listIterator();
//            StringBuilder sb = new StringBuilder();
//            while (it.hasNext()) {
//                sb.append(it.next()).append("->");
//            }
//            sb.deleteCharAt(sb.length() - 1);
//            sb.deleteCharAt(sb.length() - 1);
//            result.add(sb.toString());
//        }
//        if(root.left != null) addNode(path,root.left);
//        if(root.right != null) addNode(path,root.right);
//        path.removeLast();
//    }
    
//    leetCode82
//    public ListNode deleteDuplicates(ListNode head) {
//        ListNode newHead = new ListNode(0);
//        newHead.next = head;
//        ListNode current = newHead;
//        while (current.next != null) {
//            ListNode p = current.next;
//            int val = p.val;
//            int count = 1;
//            while (p.next != null && p.next.val == val) {
//                p = p.next;
//                count++;
//            }
//            if (count == 1) {
//                current = current.next;
//            } else {
//                current.next = p.next;
//            }
//        }
//        return newHead.next;
//    }

//    leetCode83
//    public ListNode deleteDuplicates(ListNode head) {
//        if (head == null || head.next == null) return head;
//        ListNode p1 = head;
//        ListNode p2 = head.next;
//        while (p2 != null) {
//            if (p1.val == p2.val) {
//                p1.next = p2.next;
//                p2 = p1.next;
//            } else {
//                p1 = p2;
//                p2 = p2.next;
//            }
//        }
//        return head;
//    }
    
//    LeetCode 456
//    public static void main(String[] args) {
//        int[] test = {1, 2, 3, 4};
//        System.out.println(new Solution().find132pattern(test));
//    }
//
//    public boolean find132pattern(int[] nums) {
//        if (nums.length < 3) return false;
//        LinkedList<Integer> stack = new LinkedList<>();
//        int[] minLeft = new int[nums.length];
//        minLeft[0] = Integer.MAX_VALUE;
//        for (int i = 1; i < nums.length; i++) {
//            minLeft[i] = Math.min(minLeft[i - 1], nums[i - 1]);
//        }
//        stack.addLast(nums[nums.length - 1]);
//        for (int i = nums.length - 2; i > 0; i--) {
//            int maxRight = Integer.MIN_VALUE;
//            while (stack.size() != 0 && stack.getLast() < nums[i]) {
//                maxRight = Math.max(maxRight, stack.removeLast());
//            }
//            stack.addLast(nums[i]);
//            if (maxRight > minLeft[i]) return true;
//        }
//        return false;
//    }
    
//    leetcode 287
//    public int findDuplicate(int[] nums) {
//        int val = nums[0];
//        while (nums[val] != -1) {
//            int tmp = nums[val];
//            nums[val] = -1;
//            val = tmp;
//        }
//        return val;
//    }    

//    //leetcode 452
//    public static void main(String[] args) {
//        int[][] test = {{-2147483646, -2147483645}, {2147483646, 2147483647}};
//        System.out.println(new Solution().findMinArrowShots(test));
//    }
//
//    public int findMinArrowShots(int[][] points) {
//        if (points.length == 0) return 0;
//        Arrays.sort(points, new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                if (o1[1] == o2[1]) return 0;
//                return o1[1] > o2[1] ? 1 : -1;
//            }
//        });
//        int result = 1;
//        int right = points[0][1];
//        for (int i = 1; i < points.length; i++) {
//            if (points[i][0] > right) {
//                result++;
//                right = points[i][1];
//            }
//        }
//        return result;
//    }
    
//    leetcode 137
//    public int singleNumber(int[] nums) {
//        int[] count = new int[32];
//        for (int num : nums) {
//            for (int i = 0; i < 32; i++) {
//                count[i] += num & 1;
//                num = num >>> 1;
//            }
//        }
//        int res = 0;
//        for (int i = 0; i < 32; i++) {
//            res = res | ((count[i] % 3) << i);
//        }
//        return res;
//    }
    
//    // leetcode 435
//    public int eraseOverlapIntervals(int[][] intervals) {
//        if (intervals.length == 0) return 0;
//        Arrays.sort(intervals, new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[1] - o2[1];
//            }
//        });
//        int ans = 1;
//        int right = intervals[0][1];
//        for (int i = 1; i < intervals.length; i++) {
//            if (intervals[i][0] >= right) {
//                ans++;
//                right = intervals[i][1];
//            }
//        }
//        return intervals.length - ans;
//    }
    
//     leetcode 1272
//     public List<List<Integer>> removeInterval(int[][] intervals, int[] toBeRemoved) {
//         ArrayList<List<Integer>> result = new ArrayList<>();

//         for (int i = 0; i < intervals.length; i++) {

//             int minRight = Math.min(toBeRemoved[1], intervals[i][1]);
//             int maxLeft = Math.max(toBeRemoved[0], intervals[i][0]);
//             if (maxLeft < minRight) {

//                 if (intervals[i][0] < maxLeft) {
//                     if (intervals[i][1] == minRight) {
//                         LinkedList<Integer> l = new LinkedList<>();
//                         l.addLast(intervals[i][0]);
//                         l.addLast(maxLeft);
//                         result.add(l);
//                     } else {
//                         LinkedList<Integer> l = new LinkedList<>();
//                         l.addLast(intervals[i][0]);
//                         l.addLast(maxLeft);
//                         result.add(l);
//                         LinkedList<Integer> l1 = new LinkedList<>();
//                         l1.addLast(toBeRemoved[1]);
//                         l1.addLast(intervals[i][1]);
//                         result.add(l1);
//                     }
//                 } else {
//                     if (intervals[i][1] != minRight) {
//                         LinkedList<Integer> l = new LinkedList<>();
//                         l.addLast(minRight);
//                         l.addLast(intervals[i][1]);
//                         result.add(l);
//                     }
//                 }

//             } else {
//                 LinkedList<Integer> l = new LinkedList<>();
//                 l.addLast(intervals[i][0]);
//                 l.addLast(intervals[i][1]);
//                 result.add(l);
//             }
//         }
//         return result;
//     }
    
    
//     // leetcode 57
//     public int[][] insert(int[][] intervals, int[] newInterval) {
//         int leftBound = newInterval[0];
//         int rightBound = newInterval[1];
//         boolean isFirst = true;
//         LinkedList<int[]> result = new LinkedList<>();
//         for (int i = 0; i < intervals.length; i++) {
//             if (intervals[i][1] < leftBound) {
//                 result.add(intervals[i]);
//             } else if (intervals[i][0] > rightBound) {
//                 if (isFirst) {
//                     int[] newInter = {leftBound, rightBound};
//                     result.add(newInter);
//                     isFirst = false;
//                 }
//                 result.add(intervals[i]);
//             } else {
//                 leftBound = Math.min(leftBound, intervals[i][0]);
//                 rightBound = Math.max(rightBound, intervals[i][1]);
//             }
//         }
//         if (isFirst) {
//             int[] newInter = {leftBound, rightBound};
//             result.add(newInter);
//         }
//
//         return result.toArray(new int[result.size()][]);
//     }
    
        //leetcode 986
//    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
//        if (firstList.length == 0 || secondList.length == 0) return new int[0][0];
//        ArrayList<int[]> result = new ArrayList<>();
//        int i = 0;
//        int j = 0;
//        int right = -1;
//        while (i < firstList.length || j < secondList.length) {
//            if (j >= secondList.length || (i < firstList.length && firstList[i][0] < secondList[j][0])) {
//                if (firstList[i][0] <= right) {
//                    result.add(new int[]{firstList[i][0], Math.min(right, firstList[i][1])});
//                }
//                right = Math.max(firstList[i][1], right);
//                i++;
//            } else {
//                if (secondList[j][0] <= right) {
//                    result.add(new int[]{secondList[j][0], Math.min(right, secondList[j][1])});
//                }
//                right = Math.max(right, secondList[j][1]);
//                j++;
//            }
//        }
//        return result.toArray(new int[result.size()][]);
//    }
//
//    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
//        int i = 0;
//        int j = 0;
//        ArrayList<int[]> result = new ArrayList<>();
//        while (i < firstList.length && j < secondList.length) {
//            int maxLeft = Math.max(firstList[i][0], secondList[j][0]);
//            int minRight = Math.min(firstList[i][1], secondList[j][1]);
//            if (maxLeft <= minRight) {
//                result.add(new int[]{maxLeft, minRight});
//            }
//            if (firstList[i][1] < secondList[j][1]) {
//                i++;
//            } else j++;
//        }
//        return result.toArray(new int[result.size()][]);
//    }
    
//    //leetcode 253
//    public int minMeetingRooms(int[][] intervals) {
//        Arrays.sort(intervals, new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[0] - o2[0];
//            }
//        });
//        PriorityQueue<int[]> rooms = new PriorityQueue<>(new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[1] - o2[1];
//            }
//        });
//        for (int[] inteval : intervals) {
//            if (rooms.size() == 0) {
//                rooms.add(inteval);
//                continue;
//            }
//            int[] firstEnd = rooms.peek();
//            if (firstEnd[1] <= inteval[0]) {
//                rooms.poll();
//                rooms.add(inteval);
//            } else {
//                rooms.add(inteval);
//            }
//        }
//        return rooms.size();
//
//    }
    
//    //leetcode 74
//    public boolean searchMatrix(int[][] matrix, int target) {
//        int up = 0;
//        int down = matrix.length - 1;
//        int row = -1;
//        while (up <= down) {
//            int mid = (up + down) / 2;
//            if (target >= matrix[mid][0] && target <= matrix[mid][matrix[mid].length - 1]) {
//                row = mid;
//                break;
//            } else if (target < matrix[mid][0]) {
//                down = mid - 1;
//            } else {
//                up = mid + 1;
//            }
//        }
//        if (row == -1) return false;
//        int left = 0;
//        int right = matrix[0].length - 1;
//        while (left <= right) {
//            int mid = (left + right) / 2;
//            if (matrix[row][mid] == target) return true;
//            else if (matrix[row][mid] > target) right = mid - 1;
//            else left = mid + 1;
//        }
//        return false;
//    }


    // leetcode 1047
//    public String removeDuplicates(String S) {
//        char[] array = S.toCharArray();
//        int tail = -1;
//        for (int i = 0; i < S.length(); i++) {
//            if (tail == -1 || array[i] != array[tail]) {
//                tail++;
//                array[tail] = array[i];
//            } else {
//                tail--;
//            }
//        }
//        return new String(array, 0, tail + 1);
//    }
    
//    //Leetcode 354
//    public int maxEnvelopes(int[][] envelopes) {
//        Arrays.sort(envelopes, new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                if (o1[0] == o2[0]) {
//                    return o2[1] - o1[1];
//                }
//                return o1[0] - o2[0];
//            }
//        });
//        int[] tail = new int[envelopes.length + 1];
//        int len = 1;
//        tail[len] = envelopes[0][1];
//        for (int i = 1; i < envelopes.length; i++) {
//            if (envelopes[i][1] > tail[len]) {
//                len++;
//                tail[len] = envelopes[i][1];
//                continue;
//            }
//            int left = 1;
//            int right = len;
//            while (left < right) {
//                int mid = left + (right - left) / 2;
//                if (tail[mid] < envelopes[i][1]) {
//                    left = mid + 1;
//                } else {
//                    right = mid;
//                }
//            }
//            tail[left] = Math.min(envelopes[i][1], tail[left]);
//        }
//        return len;
//    }
    
      //Leetcode 300
//    public int lengthOfLIS(int[] nums) {
//        int[] tail = new int[nums.length + 1];
//        int len = 1;
//        tail[len] = nums[0];
//        for (int i = 1; i < nums.length; i++) {
//            if (nums[i] > tail[len]) {
//                len++;
//                tail[len] = nums[i];
//                continue;
//            }
//            int left = 1;
//            int right = len;
//            while (left < right) {
//                int mid = left + (right - left) / 2;
//                if (tail[mid] < nums[i]) {
//                    left = mid + 1;
//                } else {
//                    right = mid;
//                }
//            }
//            tail[left] = Math.min(nums[i], tail[left]);
//        }
//        return len;
//    }


//
//    //Leetcode 35
//    public int searchInsert(int[] nums, int target) {
//        int left = 0;
//        int right = nums.length;
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (nums[mid] < target) {
//                left = mid + 1;
//            } else {
//                right = mid;
//            }
//        }
//        return left;
//    }
    
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode[] nodes) {
        HashSet<Integer> values = new HashSet<>();
        for (TreeNode t : nodes) {
            values.add(t.val);
        }
        return find(root, values);
    }

    public TreeNode find(TreeNode root, HashSet<Integer> values) {
        if (root == null || values.contains(root.val)) {
            return root;
        }
        TreeNode left = find(root.left, values);
        TreeNode right = find(root.right, values);
        if (left != null && right != null) {
            return root;
        }
        return left == null ? right : left;
    }

//     public Node lowestCommonAncestor(Node p, Node q) {
//         Node p1 = p;
//         Node q1 = q;
//         int pLen = 0;
//         int qLen = 0;
//         while (p1 != null) {
//             pLen++;
//             p1 = p1.parent;
//         }
//         while (q1 != null) {
//             qLen++;
//             q1 = q1.parent;
//         }
//         if (pLen > qLen) {
//             int dif = pLen - qLen;
//             for (int i = 1; i <= dif; i++) {
//                 p = p.parent;
//             }
//         } else if (pLen < qLen) {
//             int dif = qLen - pLen;
//             for (int i = 1; i <= dif; i++) {
//                 q = q.parent;
//             }
//         }
//         while (p != null && q != null) {
//             if (p == q) {
//                 return p;
//             }
//             p = p.parent;
//             q = q.parent;
//         }
//         return null;
//     }

//    public int[][] insert(int[][] intervals, int[] newInterval) {
//        int leftBound = newInterval[0];
//        int rightBound = newInterval[1];
//        boolean isFirst = true;
//        LinkedList<int[]> result = new LinkedList<>();
//        for (int i = 0; i < intervals.length; i++) {
//            if (intervals[i][1] < leftBound) {
//                result.add(intervals[i]);
//            } else if (intervals[i][0] > rightBound) {
//                if (isFirst) {
//                    int[] newInter = {leftBound, rightBound};
//                    result.add(newInter);
//                    isFirst = false;
//                }
//                result.add(intervals[i]);
//            } else {
//                leftBound = Math.min(leftBound, intervals[i][0]);
//                rightBound = Math.max(rightBound, intervals[i][1]);
//            }
//        }
//        if (isFirst) {
//            int[] newInter = {leftBound, rightBound};
//            result.add(newInter);
//        }
//        int[][] rs = new int[result.size()][2];
//        for (int i = 0; i < result.size(); i++) {
//            rs[i] = result.get(i);
//        }
//        return rs;
//    }


//    public int[][] insert(int[][] intervals, int[] newInterval) {
//        if (intervals.length == 0) {
//            int[][] rs = new int[1][2];
//            rs[0] = newInterval;
//            return rs;
//        }
//
//        int leftBound = newInterval[0];
//        int rightBound = newInterval[1];
//        int left = -2;
//        int right = -2;
//        for (int i = 0; i < intervals.length; i++) {
//            if (left == -2) {
//                if (leftBound < intervals[i][0]) {
//                    left = 2 * i - 1;
//                } else if (leftBound >= intervals[i][0] && leftBound <= intervals[i][1]) {
//                    left = 2 * i;
//                }
//            }
//            if (right == -2) {
//                if (rightBound < intervals[i][0]) {
//                    right = 2 * i - 1;
//                } else if (rightBound >= intervals[i][0] && rightBound <= intervals[i][1]) {
//                    right = 2 * i;
//                }
//            }
//        }
//        if (left == -2) left = 2 * intervals.length - 1;
//        if (right == -2) right = 2 * intervals.length - 1;
//        int l = left / 2;
//        int r = right / 2;
//        ArrayList<int[]> result = new ArrayList<>();
//        for (int i = 0; i < l; i++) {
//            result.add(intervals[i]);
//        }
//        if (leftBound >= intervals[l][0] && leftBound <= intervals[l][1]) {
//            leftBound = intervals[l][0];
//        }
//        if (leftBound > intervals[l][1]) {
//            result.add(intervals[l]);
//        }
//
//        if (rightBound < intervals[r][0]) {
//            r = r - 1;
//        }
//        if (rightBound >= intervals[r][0] && rightBound <= intervals[r][1]) {
//            rightBound = intervals[r][1];
//        }
//        int[] iv = {leftBound, rightBound};
//        result.add(iv);
//        for (int i = r + 1; i < intervals.length; i++) {
//            result.add(intervals[i]);
//        }
//
//        int[][] rs = new int[result.size()][2];
//        for (int i = 0; i < result.size(); i++) {
//            rs[i] = result.get(i);
//        }
//        return rs;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().isUnique("leet"));
//    }
//
//    public boolean isUnique(String astr) {
//        int mask = 0;
//        for (int i = 0; i < astr.length(); i++) {
//            int c = astr.charAt(i) - 'a';
//            if ((mask & (1 << c)) != 0) return false;
//            mask = mask | (1 << c);
//        }
//        return true;
//    }

//    TreeNode result = null;
//
//    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        find(root, p, q);
//        return result;
//    }
//
//    public boolean find(TreeNode root, TreeNode p, TreeNode q) {
//        if (root == null) return false;
//        if (root == p || root == q) {
//            if (find(root.left, p, q) || find(root.right, p, q)) {
//                result = root;
//            }
//            return true;
//        }
//        boolean left = find(root.left, p, q);
//        boolean right = find(root.right, p, q);
//        if (left && right) result = root;
//        return left || right;
//    }

//    public int longestConsecutive(int[] nums) {
//        HashSet<Integer> set = new HashSet<>();
//        for (int n : nums) {
//            set.add(n);
//        }
//        int result = 0;
//        for (int n : nums) {
//            if (set.contains(n - 1)) continue;
//            int len = 1;
//            int next = n + 1;
//            while (set.contains(next)) {
//                next++;
//                len++;
//            }
//            result = Math.max(result, len);
//        }
//        return result;
//    }


//    public int[] countBits(int num) {
//        int[] result = new int[num + 1];
//        result[0] = 0;
//        if (num == 0) return result;
//        result[1] = 1;
//        if (num == 1) return result;
//        int currentSize = 2;
//        int index = 2;
//        while (index <= num) {
//            for (int i = 0; i < currentSize; i++) {
//                if (index > num) return result;
//                result[index] = result[i] + 1;
//                index++;
//            }
//            currentSize *= 2;
//        }
//        return result;
//    }

//    public List<List<Integer>> subsets(int[] nums) {
//        LinkedList<List<Integer>> result = new LinkedList<>();
//        result.add(new LinkedList<>());
//        for (int n : nums) {
//            int len = result.size();
//            for (int i = 0; i < len; i++) {
//                List<Integer> l = result.get(i);
//                LinkedList<Integer> newL = (LinkedList<Integer>) ((LinkedList<Integer>) l).clone();
//                newL.add(n);
//                result.add(newL);
//            }
//        }
//        return result;
//    }

//    public boolean isMonotonic(int[] A) {
//        int increase = 0;
//        for (int i = 1; i < A.length; i++) {
//            if (A[i] > A[i - 1]) {
//                if (increase == 0) {
//                    increase = 1;
//                } else if (increase == -1) {
//                    return false;
//                }
//            }
//            if (A[i] < A[i - 1]) {
//                if (increase == 0) {
//                    increase = -1;
//                } else if (increase == 1) {
//                    return false;
//                }
//            }
//        }
//        return true;
//    }

//    public static void main(String[] args) {
//        int[] test = {2,1,5,6,2,3};
//        System.out.println(new Solution().largestRectangleArea(test));
//    }
//
//    public int largestRectangleArea(int[] heights) {
//        int[] leftMin = new int[heights.length];
//        int[] rightMin = new int[heights.length];
//        LinkedList<Integer> stack = new LinkedList<>();
//        for (int i = 0; i < heights.length; i++) {
//            while (stack.size() != 0 && heights[stack.getLast()] >= heights[i]) {
//                int index = stack.removeLast();
//                rightMin[index] = i;
//            }
//            if (stack.size() > 0) {
//                leftMin[i] = stack.getLast();
//            } else {
//                leftMin[i] = -1;
//            }
//            stack.addLast(i);
//        }
//        int result = 0;
//        for (int i = 0; i < heights.length; i++) {
//            int right = rightMin[i] == 0 ? heights.length : rightMin[i];
//            int area = heights[i] * (right - leftMin[i] - 1);
//            result = Math.max(area, result);
//        }
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {1,0, 0, 0};
//        int[] n2 = {3, 10, 11};
//        new Solution().merge(test,1,n2,3);
//        for (int i : test) {
//            System.out.println(i);
//        }
//    }

//    public void merge(int[] nums1, int m, int[] nums2, int n) {
//        int i = m - 1;
//        int j = n - 1;
//        int pos = nums1.length - 1;
//        while (pos >= 0) {
//            if (j < 0 || (i >= 0 && nums1[i] > nums2[j])) {
//                nums1[pos] = nums1[i];
//                pos--;
//                i--;
//            } else {
//                nums1[pos] = nums2[j];
//                pos--;
//                j--;
//            }
//        }
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().longestSubstring("bbaaacbd", 3));
//    }
//
//    public int longestSubstring(String s, int k) {
//        return findLongest(s, 0, s.length() - 1, k);
//    }
//
//    public int findLongest(String s, int left, int right, int k) {
//        if (right - left + 1 < k) return 0;
//        int[] count = new int[26];
//        for (int i = left; i <= right; i++) {
//            count[s.charAt(i) - 'a']++;
//        }
//        int l = left;
//        int r = left;
//        boolean isAnswer = true;
//        int result = 0;
//        while (r <= right) {
//            if (count[s.charAt(r) - 'a'] < k) {
//                isAnswer = false;
//                int len = findLongest(s, l, r - 1, k);
//                result = Math.max(len, result);
//                r++;
//                l = r;
//            } else {
//                r++;
//            }
//        }
//        if (isAnswer) result = right - left + 1;
//        else {
//            int len = findLongest(s, l, r - 1, k);
//            result = Math.max(result, len);
//        }
//        return result;
//    }


//    public static void main(String[] args) {
//        int[][] test = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
//        System.out.println(new Solution().spiralOrder(test));
//    }
//
//    public List<Integer> spiralOrder(int[][] matrix) {
//        LinkedList<Integer> result = new LinkedList<>();
//        int row = matrix.length;
//        int col = matrix[0].length;
//        int count = 0;
//        int i = 0;
//        int j = 0;
//        while (2 * count < row && 2 * count < col) {
//            while (j < col - count) {
//                result.add(matrix[i][j]);
//                j++;
//            }
//            j--;
//            i++;
//            while (i < row - count) {
//                result.add(matrix[i][j]);
//                i++;
//            }
//            i--;
//            j--;
//            if (2 * count + 1 >= col || 2 * count + 1 >= row)
//                break;
//            while (j >= count) {
//                result.add(matrix[i][j]);
//                j--;
//            }
//            j++;
//            i--;
//            while (i > count) {
//                result.add(matrix[i][j]);
//                i--;
//            }
//            i++;
//            j++;
//            count++;
//        }
//
//        return result;
//    }

//    public int[][] transpose(int[][] matrix) {
//        int[][] result = new int[matrix[0].length][matrix.length];
//        for (int i = 0; i < matrix.length; i++) {
//            for (int j = 0; j < matrix[i].length; j++) {
//                result[j][i] = matrix[i][j];
//            }
//        }
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {10, 1, 2, 4, 7, 2};
//        System.out.println(new Solution().longestSubarray(test, 5));
//    }
//
//    public int longestSubarray(int[] nums, int limit) {
//        LinkedList<Integer> minQueue = new LinkedList<>();
//        LinkedList<Integer> maxQueue = new LinkedList<>();
//        int left = 0;
//        int right = 0;
//        int result = 0;
//        while (right < nums.length) {
//            while (minQueue.size()!=0 && nums[right] < minQueue.getLast()) {
//                minQueue.removeLast();
//            }
//            while (maxQueue.size()!=0 && nums[right] > maxQueue.getLast()) {
//                maxQueue.removeLast();
//            }
//            minQueue.addLast(nums[right]);
//            maxQueue.addLast(nums[right]);
//            right++;
//            if (maxQueue.getFirst() - minQueue.getFirst() > limit) {
//                if (minQueue.getFirst() == nums[left]) {
//                    minQueue.removeFirst();
//                }
//                if (maxQueue.getFirst() == nums[left]) {
//                    maxQueue.removeFirst();
//                }
//                left++;
//            } else {
//                result = Math.max(result,right-left);
//            }
//        }
//        return result;
//    }
//
//    public int longestSubarray(int[] nums, int limit) {
//        TreeMap<Integer, Integer> count = new TreeMap<>();
//        int left = 0;
//        int right = 0;
//        int result = 0;
//        while (right < nums.length) {
//            if (count.containsKey(nums[right])) {
//                count.put(nums[right], count.get(nums[right]) + 1);
//            } else {
//                count.put(nums[right], 1);
//            }
//            right++;
//            if (count.lastKey() - count.firstKey() > limit) {
//                count.put(nums[left], count.get(nums[left]) - 1);
//                if (count.get(nums[left]) == 0) count.remove(nums[left]);
//                left++;
//            } else {
//                result = Math.max(result, right - left);
//            }
//        }
//        return result;
//    }


//    public int[][] flipAndInvertImage(int[][] A) {
//        for (int i = 0; i < A.length; i++) {
//            int l = 0;
//            int r = A[i].length - 1;
//            while (r > l) {
//                int tmp = A[i][l];
//                A[i][l] = A[i][r] ^ 1;
//                A[i][r] = tmp ^ 1;
//                l++;
//                r--;
//            }
//            if (l == r) A[i][r] = A[i][r] ^ 1;
//        }
//        return A;
//    }

//    public static void main(String[] args) {
//
//    }
//
//    public int maxSatisfied(int[] customers, int[] grumpy, int X) {
//        int sum = 0;
//        for (int i = 0; i < customers.length; i++) {
//            if (grumpy[i] == 0) sum += customers[i];
//        }
//        int max = 0;
//        int now = 0;
//        for (int i = 0; i < X; i++) {
//            if (grumpy[i] == 1) now += customers[i];
//        }
//        max = now;
//        for (int i = 1; i + X - 1 < customers.length; i++) {
//            if (grumpy[i - 1] == 1) now -= customers[i - 1];
//            if (grumpy[i + X - 1] == 1) now += customers[i + X - 1];
//            max = Math.max(now, max);
//        }
//        return max + sum;
//    }

//    public int findShortestSubArray(int[] nums) {
//        HashMap<Integer, Integer> count = new HashMap<>();
//        int max = 0;
//        for (int n : nums) {
//            if (count.containsKey(n)) {
//                count.put(n, count.get(n) + 1);
//            } else {
//                count.put(n, 1);
//            }
//            max = Math.max(max, count.get(n));
//        }
//
//        HashMap<Integer, Integer> pos = new HashMap<>();
//        for (int key : count.keySet()) {
//            if (count.get(key) == max) {
//                pos.put(key, -1);
//            }
//        }
//
//        int size = pos.size();
//        for (int i = 0; i < nums.length; i++) {
//            if (size == 0) break;
//            if (pos.containsKey(nums[i]) && pos.get(nums[i]) == -1) {
//                pos.put(nums[i], i);
//                size--;
//            }
//        }
//
//        int result = Integer.MAX_VALUE;
//        for (int i = nums.length - 1; i >= 0; i--) {
//            if (pos.size() == 0) break;
//            if (pos.containsKey(nums[i])) {
//                int len = i - pos.get(nums[i]) + 1;
//                result = Math.min(result, len);
//                pos.remove(nums[i]);
//            }
//        }
//
//        return result;
//
//    }

//    public static void main(String[] args) {
//        int[] test = {1,2,3};
//        System.out.println(new Solution().jump(test));
//    }
//
//    public int jump(int[] nums) {
//        int cureentStep = 0;
//        int currentEdge = 0;
//        int nextEdge = 0;
//        for (int i = 0; i < nums.length - 1; i++) {
//            if (i > currentEdge) {
//                cureentStep++;
//                currentEdge = nextEdge;
//            }
//            if (nums[i] + i > nextEdge) nextEdge = nums[i] + i;
//            if (nextEdge >= nums.length - 1) return cureentStep + 1;
//        }
//        return cureentStep;
//    }

//    public static void main(String[] args) {
//        int[] test = {1,1,1,0,0,0,1,1,1,1,0};
//        System.out.println(new Solution().longestOnes(test,2));
//    }
//
//    public int longestOnes(int[] A, int K) {
//        int rest = K;
//        int left = 0;
//        int right = 0;
//        int result = 0;
//        while (right < A.length) {
//            if (A[right] == 0) {
//                rest--;
//            }
//            right++;
//            if (rest < 0) {
//                if (A[left] == 0) {
//                    rest++;
//                }
//                left++;
//            } else {
//                result = Math.max(result, right - left);
//            }
//        }
//        return result;
//    }

//    public int findMaxConsecutiveOnes(int[] nums) {
//        int rest = 1;
//        int left = 0;
//        int right = 0;
//        int result = 0;
//        while (right < nums.length) {
//            if (nums[right] == 0) {
//                rest--;
//            }
//            right++;
//            if (rest < 0) {
//                if (nums[left] == 0) {
//                    rest++;
//                }
//                left++;
//            } else {
//                result = Math.max(result, right - left);
//            }
//        }
//        return result;
//    }


//    public int firstMissingPositive(int[] nums) {
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] == -1) nums[i] = -2;
//        }
//
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] > 0 && nums[i] <= nums.length) {
//                int pos = nums[i] - 1;
//                int val = nums[pos];
//                while (val > 0 && val <= nums.length) {
//                    nums[pos] = -1;
//                    pos = val - 1;
//                    val = nums[pos];
//                }
//                nums[pos] = -1;
//            }
//        }
//
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] != -1) return i + 1;
//        }
//        return nums.length + 1;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().addStrings("123", "123"));
//    }

//    public String addStrings(String num1, String num2) {
//        int i = 0;
//        int j = 0;
//        int next = 0;
//        StringBuilder sb = new StringBuilder();
//        while (i < num1.length() || j < num2.length()) {
//            int digit1 = 0;
//            int digit2 = 0;
//            if (i < num1.length()) {
//                digit1 = num1.charAt(num1.length() - i - 1) - '0';
//            }
//            if (j < num2.length()) {
//                digit2 = num2.charAt(num2.length() - j - 1) - '0';
//            }
//            next = digit1 + digit2 + next;
//            sb.append((char) ('0' + next % 10));
//            next = next / 10;
//            i++;
//            j++;
//        }
//        if (next == 1) sb.append('1');
//        return sb.reverse().toString();
//    }

//    public String addStrings(String num1, String num2) {
//        StringBuilder sb = new StringBuilder();
//        if (num1.length() < num2.length()) {
//            String tmp = num1;
//            num1 = num2;
//            num2 = tmp;
//        }
//        int next = 0;
//
//        for (int i = 0; i < num1.length(); i++) {
//            if (i < num2.length()) {
//                int digit1 = num1.charAt(num1.length() - 1 - i) - '0';
//                int digit2 = num2.charAt(num2.length() - 1 - i) - '0';
//                int sum = digit1 + digit2 + next;
//                if (sum >= 10) {
//                    sum = sum - 10;
//                    next = 1;
//                } else {
//                    next = 0;
//                }
//                sb.append((char)('0' + sum));
//            } else {
//                int digit1 = num1.charAt(num1.length() - 1 - i) - '0';
//                int sum = digit1 + next;
//                if (sum >= 10) {
//                    sum = sum - 10;
//                    next = 1;
//                } else {
//                    next = 0;
//                }
//                sb.append((char)('0' + sum));
//            }
//        }
//        if (next == 1) sb.append(next);
//
//        sb.reverse();
//        return sb.toString();
//    }

//    public static void main(String[] args) {
//        int[] test = {0,0,0,1,0,1,1,0};
//        System.out.println(new Solution().minKBitFlips(test, 3));
//    }

//    public int minKBitFlips(int[] A, int K) {
//        LinkedList<Integer> queue = new LinkedList<>();
//        int count = 0;
//        for (int i = 0; i < A.length; i++) {
//            if (queue.size() > 0 && queue.getFirst() + K - 1 < i) {
//                queue.removeFirst();
//            }
//            if (queue.size() % 2 == A[i]) {
//                if (i + K - 1 >= A.length) return -1;
//                count++;
//                queue.add(i);
//            }
//        }
//        return count;
//    }

//    public boolean checkInclusion(String s1, String s2) {
//        int[] ori = new int[26];
//        int[] count = new int[26];
//        for (int i = 0; i < s1.length(); i++) {
//            ori[s1.charAt(i) - 'a']++;
//        }
//        int left = 0;
//        int right = 0;
//        while (right < s2.length()) {
//            int r = s2.charAt(right) - 'a';
//            right++;
//            count[r]++;
//            while (count[r] > ori[r]) {
//                int l = s2.charAt(left) - 'a';
//                count[l]--;
//                left++;
//            }
//            if (right - left == s1.length()) return true;
//        }
//        return false;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().findAnagrams("cbaebabacd", "abc"));
//    }

//    public List<Integer> findAnagrams(String s, String p) {
//        ArrayList<Integer> result = new ArrayList<>();
//        int[] origin = new int[26];
//        int[] count = new int[26];
//
//        for (int i = 0; i < p.length(); i++) {
//            origin[p.charAt(i) - 'a']++;
//        }
//
//        int left = 0;
//        int right = 0;
//        while (right < s.length()) {
//            int r = s.charAt(right) - 'a';
//            right++;
//            count[r]++;
//
//            while (count[r] > origin[r]) {
//                count[s.charAt(left)-'a']--;
//                left++;
//            }
//            if (right - left == p.length()) {
//                result.add(left);
//            }
//        }
//        return result;
//    }


//    HashMap<Character, Integer> count = new HashMap<>();
//
//    public List<Integer> findAnagrams(String s, String p) {
//        ArrayList<Integer> result = new ArrayList<>();
//        if(s.length() < p.length()) return result;
//        for (int i = 0; i < p.length(); i++) {
//            if (count.containsKey(p.charAt(i))) {
//                count.put(p.charAt(i), count.get(p.charAt(i)) + 1);
//            } else {
//                count.put(p.charAt(i), 1);
//            }
//        }
//        int left = 0;
//        int right = p.length() - 1;
//        for (int i = 0; i <= right; i++) {
//            if (count.containsKey(s.charAt(i))) {
//                count.put(s.charAt(i), count.get(s.charAt(i)) - 1);
//            }
//        }
//        while (right < s.length()) {
//            if (check()) {
//                result.add(left);
//            }
//            if (count.containsKey(s.charAt(left))) {
//                count.put(s.charAt(left), count.get(s.charAt(left)) + 1);
//            }
//            left++;
//            right++;
//            if (right == s.length()) break;
//            if (count.containsKey(s.charAt(right))) {
//                count.put(s.charAt(right), count.get(s.charAt(right)) - 1);
//            }
//
//        }
//
//        return result;
//    }
//
//    public boolean check() {
//        for (Character c : count.keySet()) {
//            if (count.get(c) != 0) return false;
//        }
//        return true;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().minWindow("bba", "ab"));
//    }

//    HashMap<Character, Integer> origin = new HashMap<>();
//    HashMap<Character, Integer> count = new HashMap<>();
//
//
//    public String minWindow(String s, String t) {
//        String result = ";
//        for (int i = 0; i < t.length(); i++) {
//            if (origin.containsKey(t.charAt(i))) {
//                origin.put(t.charAt(i), origin.get(t.charAt(i)) + 1);
//            } else {
//                origin.put(t.charAt(i), 1);
//            }
//        }
//        int left = 0;
//        int right = -1;
//
//        while (right < s.length()) {
//            while (!compare()) {
//                right++;
//                if(right == s.length()) break;
//                if (origin.containsKey(s.charAt(right))) {
//                    if (count.containsKey(s.charAt(right))) {
//                        count.put(s.charAt(right), count.get(s.charAt(right)) + 1);
//                    } else {
//                        count.put(s.charAt(right), 1);
//                    }
//                }
//            }
//            if(right == s.length()) break;
//            String sub = s.substring(left, right + 1);
//            if (result.length() == 0 || result.length() > sub.length()) result = sub;
//            if (count.containsKey(s.charAt(left))) {
//                count.put(s.charAt(left),count.get(s.charAt(left))-1);
//            }
//            left++;
//
//        }
//
//        return result;
//
//    }
//
//    public boolean compare() {
//        Iterator iter = origin.entrySet().iterator();
//        while (iter.hasNext()) {
//            Map.Entry entry = (Map.Entry) iter.next();
//            Character key = (Character) entry.getKey();
//            Integer val = (Integer) entry.getValue();
//            if (count.getOrDefault(key, 0) < val) {
//                return false;
//            }
//        }
//        return true;
//
//    }


//    public static void main(String[] args) {
//        int[] test = {4, 3, 2, 7, 8, 2, 3, 1};
//        System.out.println(new Solution().findDisappearedNumbers(test));
//
//    }
//
//    public List<Integer> findDisappearedNumbers(int[] nums) {
//        ArrayList<Integer> result = new ArrayList<Integer>();
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] == -1) continue;
//            int value = nums[i];
//            while (nums[value - 1] != -1) {
//                int tmp = nums[value - 1];
//                nums[value - 1] = -1;
//                value = tmp;
//            }
//        }
//        for (int i = 0; i < nums.length; i++) {
//            if(nums[i]!=-1)
//                result.add(i+1);
//
//        }
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {4,8,12,16};
//        System.out.println(new Solution().maxTurbulenceSize(test));
//    }
//
//    public int maxTurbulenceSize(int[] arr) {
//        int result = 0;
//        int left = 0;
//        int right = 1;
//        while (right < arr.length) {
//            boolean is_increase = arr[right] > arr[left];
//            while (right < arr.length && arr[right] != arr[right - 1] && is_increase == (arr[right] > arr[right - 1])) {
//                right++;
//                is_increase = !is_increase;
//            }
//            result = Math.max(result, right - left);
//            if (right == arr.length) {
//                break;
//            } else if (arr[right] == arr[right - 1]) {
//                left = right;
//                right = right+1;
//            } else {
//                left = right-1;
//            }
//        }
//        result = Math.max(result, right - left);
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {-1,4,2,3};
//        System.out.println(new Solution().checkPossibility(test));
//
//    }
//
//    public boolean checkPossibility(int[] nums) {
//        int index = -1;
//        for (int i = 0; i < nums.length - 1; i++) {
//            if (nums[i] > nums[i + 1]) {
//                if (index == -1) index = i;
//                else return false;
//            }
//        }
//        if (index == -1 || index == 0 || index == nums.length-2) return true;
//
//        if(nums[index]<=nums[index+2] || nums[index-1]<=nums[index+1]) return true;
//
//        return false;
//
//
//    }


//    public int equalSubstring(String s, String t, int maxCost) {
//        int rest = maxCost;
//        int left = 0;
//        int right = 0;
//        int result = 0;
//        while (right < s.length()) {
//            rest = rest - Math.abs(s.charAt(right) - t.charAt(right));
//            right++;
//            if (rest < 0) {
//                rest = rest + Math.abs(s.charAt(left) - t.charAt(left));
//                left++;
//            } else {
//                result = Math.max(result, right - left);
//            }
//        }
//        return result;
//    }

//    public double findMaxAverage(int[] nums, int k) {
//        int left = 0;
//        int right = k - 1;
//        int sum = 0;
//        for (int i = 0; i < k; i++) {
//            sum += nums[i];
//        }
//        int result = sum;
//        while (right < nums.length - 1) {
//            sum -= nums[left];
//            left++;
//            right++;
//            sum += nums[right];
//            result = Math.max(sum, result);
//        }
//        return (double)result / k;
//    }

//    public static void main(String[] args) {
//        int[] test = {1,4,4};
//        System.out.println(new Solution().minSubArrayLen(4, test));
//    }


//    public int minSubArrayLen(int target, int[] nums) {
//        int sum = 0;
//        int result = Integer.MAX_VALUE;
//        int left = 0;
//        int right = 0;
//        while (right < nums.length) {
//            sum = sum + nums[right];
//            if (sum < target) {
//                right++;
//            } else {
//                while (sum >= target) {
//                    result = Math.min(result, right - left + 1);
//                    sum = sum - nums[left];
//                    left++;
//                }
//                right++;
//            }
//        }
//        if (result == Integer.MAX_VALUE) return -1;
//        else return result;
//    }

//    public static void main(String[] args) {
//        String s = "AABCABBB";
//        System.out.println(new Solution().characterReplacement(s, 2));
//    }
//
//    public int characterReplacement(String s, int k) {
//        if (s.length() < 2) return s.length();
//        int result = 0;
//        int[] freq = new int[26];
//        int left = 0;
//        int right = 0;
//        int max = 0;
//        while (right < s.length()) {
//
//            freq[s.charAt(right) - 'A']++;
//            max = Math.max(max, freq[s.charAt(right) - 'A']);
//            right++;
//            int len = right - left;
//            if (len - max > k) {
//                freq[s.charAt(left) - 'A']--;
//                left++;
//
//            } else {
//                result = Math.max(result, len);
//            }
//        }
//        return result;
//    }


//    public int characterReplacement(String s, int k) {
//        if (s.length() == 0) return 0;
//        int[] freq = new int[26];
//        int result = 0;
//        int left = 0;
//        int right = 0;
//        int max = 1;
//        freq[s.charAt(0) - 'A']++;
//        while (left < s.length()) {
//            int len = right - left + 1;
//            if ((right == s.length() - 1) && (len) <= result) {
//                break;
//            }
//            if (len - max <= k) {
//                result = Math.max(result, len);
//                if (right < s.length() - 1) {
//                    right++;
//                    freq[s.charAt(right) - 'A']++;
//                    max = Math.max(freq[s.charAt(right) - 'A'], max);
//                }
//            } else {
//                freq[s.charAt(left) - 'A']--;
//                max--;
//                left++;
//                if (right < s.length() - 1) {
//                    right++;
//                    freq[s.charAt(right) - 'A']++;
//                    max = Math.max(freq[s.charAt(right) - 'A'], max);
//                }
//            }
//        }
//        return result;
//    }


//    public static void main(String[] args) {
//        String s = "(()()";
//        System.out.println(new Solution().longestValidParentheses(s));
//    }
//
//    public int longestValidParentheses(String s) {
//        int result = 0;
//        int left = 0;
//        int right = 0;
//        for (int i = 0; i < s.length(); i++) {
//            if (s.charAt(i) == ')') {
//                right++;
//            } else {
//                left++;
//            }
//            if (right > left) {
//                right = 0;
//                left = 0;
//            }
//            if (left == right) {
//                result = Math.max(result, 2 * left);
//            }
//        }
//
//        left = 0;
//        right = 0;
//        for (int i = s.length() - 1; i >= 0; i--) {
//            if (s.charAt(i) == ')') {
//                right++;
//            } else {
//                left++;
//            }
//            if (left > right) {
//                right = 0;
//                left = 0;
//            }
//            if (left == right) {
//                result = Math.max(result, 2 * left);
//            }
//        }
//        return result;
//    }


//    public static void main(String[] args) {
//        String s = "(()()";
//        System.out.println(new Solution().longestValidParentheses(s));
//    }
//
//    public int longestValidParentheses(String s) {
//        LinkedList<Integer> stack = new LinkedList<>();
//        int lastRight = -1;
//        int result = 0;
//        for (int i = 0; i < s.length(); i++) {
//            if (s.charAt(i) == '(') {
//                stack.push(i);
//            } else {
//                if (stack.size() > 0) {
//                    int index = stack.pop();
//                    int len;
//                    if (stack.size() == 0) {
//                        len = i - lastRight;
//                    } else {
//                        len = i - stack.getFirst();
//                    }
//                    if (len > result) {
//                        result = len;
//                    }
//                } else {
//                    lastRight = i;
//                }
//            }
//        }
//        return result;
//    }


//    public int longestValidParentheses(String s) {
//        int max = 0;
//        int[] dp = new int[s.length()];
//        for (int i = 1; i < dp.length; i++) {
//            if (s.charAt(i) == ')') {
//                if (s.charAt(i - 1) == '(') {
//                    if (i > 1) {
//                        dp[i] = 2 + dp[i - 2];
//                    } else {
//                        dp[i] = 2;
//                    }
//                } else {
//                    if (i - 1 - dp[i - 1] >= 0 && s.charAt(i - 1 - dp[i - 1]) == '(') {
//                        if (i - 1 - dp[i - 1] > 0) {
//                            dp[i] = 2 + dp[i - 1] + dp[i - 2 - dp[i - 1]];
//                        } else {
//                            dp[i] = 2 + dp[i - 1];
//                        }
//
//                    }
//                }
//            }
//            if (dp[i] > max) {
//                max = dp[i];
//            }
//        }
//        return max;
//
//    }


//    public int[] fairCandySwap(int[] A, int[] B) {
//        int[] result = new int[2];
//        HashSet<Integer> set = new HashSet<>();
//        int sumA = 0;
//        int sumB = 0;
//        for (int k : A) {
//            sumA += k;
//            set.add(k);
//        }
//        for (int j : B) {
//            sumB += j;
//        }
//        int dif = sumA - sumB;
//        dif = dif / 2;
//        for (int j : B) {
//            if (set.contains(j + dif)) {
//                result[0] = j + dif;
//                result[1] = j;
//                break;
//            }
//
//        }
//
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {-1,-1,-1,0,1,1};
//        System.out.println(new Solution().pivotIndex(test));
//    }
//
//    public int pivotIndex(int[] nums) {
//        int sum = 0;
//        for (int i = 0; i < nums.length; i++) {
//            sum += nums[i];
//        }
//        int sumLeft = 0;
//        for (int i = 0; i < nums.length - 1; i++) {
//            if ((sum - nums[i]) == 2 * sumLeft) {
//                return i;
//            }
//            sumLeft += nums[i];
//        }
//        return -1;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().countAndSay(4));
//    }

//    public String countAndSay(int n) {
//        StringBuilder result = new StringBuilder("1");
//        for (int j = 2; j <= n; j++) {
//            StringBuilder sb = new StringBuilder(");
//            char current = result.charAt(0);
//            int count = 1;
//            for (int i = 1; i < result.length(); i++) {
//                if (result.charAt(i) != current) {
//                    sb.append(count).append(current);
//                    current = result.charAt(i);
//                    count = 1;
//                } else {
//                    count++;
//                }
//            }
//            sb.append(count).append(current);
//            result = sb;
//        }
//        return result.toString();
//    }


//    public int numEquivDominoPairs(int[][] dominoes) {
//        int nums[] = new int[100];
//        int result = 0;
//
//        for (int i = 0; i < dominoes.length; i++) {
//            int a = dominoes[i][0];
//            int b = dominoes[i][1];
//            if (a < b) {
//                int tmp = a;
//                a = b;
//                b = tmp;
//            }
//            int sum = a * 10 + b;
//            result += nums[sum];
//            nums[sum]++;
//        }
//
//        return result;
//    }

//    public int numEquivDominoPairs(int[][] dominoes) {
//        HashMap<Domino, Integer> map = new HashMap<>();
//        for (int i = 0; i < dominoes.length; i++) {
//            Domino d = new Domino(dominoes[i][0], dominoes[i][1]);
//            if (map.containsKey(d)) {
//                map.put(d, map.get(d) + 1);
//            } else {
//                map.put(d, 1);
//            }
//        }
//        int result = 0;
//        for (Domino key : map.keySet()) {
//            int value = map.get(key);
//            result += value * (value - 1) / 2;
//        }
//        return result;
//    }
//
//    static class Domino {
//        int a;
//        int b;
//
//        public Domino(int a, int b) {
//            this.a = a;
//            this.b = b;
//        }
//
//        @Override
//        public boolean equals(Object o) {
//            if (this == o) return true;
//            if (o == null || getClass() != o.getClass()) return false;
//            Domino domino = (Domino) o;
//            return (a == domino.a && b == domino.b) || (a == domino.b && b == domino.a);
//        }
//
//        @Override
//        public int hashCode() {
//            return Objects.hash(a + b, a * b);
//        }
//    }


//    public boolean isValidSudoku(char[][] board) {
//        ArrayList<HashSet<Character>> rows = new ArrayList<>();
//        ArrayList<HashSet<Character>> columns = new ArrayList<>();
//        ArrayList<HashSet<Character>> areas = new ArrayList<>();
//
//        for (int i = 0; i < 9; i++) {
//            rows.add(new HashSet<>());
//            columns.add(new HashSet<>());
//            areas.add(new HashSet<>());
//        }
//
//        for (int i = 0; i < 9; i++) {
//            for (int j = 0; j < 9; j++) {
//                if (board[i][j] != '.') {
//                    char c = board[i][j];
//                    int area = j / 3 + (i / 3) * 3;
//                    if (rows.get(i).contains(c) || columns.get(j).contains(c) || areas.get(area).contains(c))
//                        return false;
//                    rows.get(i).add(c);
//                    columns.get(j).add(c);
//                    areas.get(area).add(c);
//                }
//            }
//        }
//
//        return true;
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().findLengthOfLCIS(new int[]{2,2,2,2,2,2}));
//    }
//
//
//    public int findLengthOfLCIS(int[] nums) {
//        if (nums.length == 0 || nums.length == 1) return nums.length;
//        int result = 0;
//        int i = 0;
//        int j = 1;
//        while (j < nums.length) {
//            if (nums[j] > nums[j - 1]) {
//
//            } else {
//                int len = j - i;
//                if (len > result)
//                    result = len;
//                i = j;
//            }
//            j++;
//        }
//        int len = j - i;
//        if (len > result)
//            result = len;
//
//        return result;
//    }


//    public int maximumProduct(int[] nums) {
//        int max = Integer.MIN_VALUE;
//        int maxIndex = -1;
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] > max) {
//                max = nums[i];
//                maxIndex = i;
//            }
//        }
//
//        int second = Integer.MIN_VALUE;
//        int secondIndex = -1;
//        for (int i = 0; i < nums.length; i++) {
//            if (i == maxIndex) {
//                continue;
//            }
//            if (nums[i] > second) {
//                second = nums[i];
//                secondIndex = i;
//            }
//        }
//
//        int third = Integer.MIN_VALUE;
//        for (int i = 0; i < nums.length; i++) {
//            if (i == maxIndex || i == secondIndex) continue;
//            if (nums[i] > third) {
//                third = nums[i];
//            }
//        }
//
//        int min = Integer.MAX_VALUE;
//        int minIndex = -1;
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] < min) {
//                min = nums[i];
//                minIndex = i;
//            }
//        }
//
//        int min2 = Integer.MAX_VALUE;
//        for (int i = 0; i < nums.length; i++) {
//            if (i == minIndex) continue;
//            if (nums[i] < min2) {
//                min2 = nums[i];
//            }
//        }
//
//        return Math.max(max * min * min2, max * second * third);
//
//    }

//    int result = 0;
//
////    public static void main(String[] args) {
////        int[] test = {2147483647,2147483647,2147483647,2147483647,2147483647,2147483647};
////        System.out.println(new Solution().reversePairs(test));
////    }
//
//    public int reversePairs(int[] nums) {
//        mergeSort(nums, 0, nums.length - 1);
//        return result;
//    }
//
//    public void mergeSort(int[] nums, int left, int right) {
//        if (left < right) {
//            int middle = (left + right) / 2;
//            mergeSort(nums, left, middle);
//            mergeSort(nums, middle + 1, right);
//            merge(nums, left, right);
//        }
//    }
//
//    public void merge(int[] nums, int left, int right) {
//        int[] tmp = new int[right - left + 1];
//        int middle = (left + right) / 2;
//
//        int i = left;
//        for (int j = middle + 1; j <= right; j++) {
//            while (i <= middle && nums[i] <= (long)2 * nums[j]) {
//                i++;
//            }
//            if (i > middle) break;
//            result += middle - i + 1;
//        }
//
//        i = left;
//        int j = middle + 1;
//        int count = 0;
//
//        while (i <= middle || j <= right) {
//            if (j > right || (i<=middle && (nums[i] < nums[j]))) {
//                tmp[count] = nums[i];
//                i++;
//            } else {
//                tmp[count] = nums[j];
//                j++;
//            }
//            count++;
//        }
//
//        if (tmp.length >= 0) System.arraycopy(tmp, 0, nums, left, tmp.length);
//
//    }


//    public int[] smallestK(int[] arr, int k) {
//        int[] result = new int[k];
//        quickSort(arr, 0, arr.length - 1, k);
//        System.arraycopy(arr, 0, result, 0, k);
//        return result;
//    }
//
//    public void quickSort(int[] array, int left, int right, int k) {
//        if (left < right) {
//            int i = left;
//            int j = right;
//            int pivot = array[i];
//            while (i < j) {
//                while (i < j && array[j] > pivot) {
//                    j--;
//                }
//                if (i < j) {
//                    array[i] = array[j];
//                    i++;
//                }
//                while (i < j && array[i] <= pivot) {
//                    i++;
//                }
//                if (i < j) {
//                    array[j] = array[i];
//                    j--;
//                }
//            }
//            array[i] = pivot;
//
//            if (i + 1 > k) {
//                quickSort(array, left, i - 1, k);
//            }
//
//            if (i + 1 < k) {
//                quickSort(array, i + 1, right, k);
//            }
//        }
//    }


//    public static void main(String[] args) {
//        System.out.println(new Solution().countPrimes(3));
//    }
//
//    public int countPrimes(int n) {
//        if (n < 3) return 0;
//        int result = 0;
//        int[] isPrime = new int[n];
//        for (int i = 2; i < n; i++) {
//            if (isPrime[i] == 0) {
//                result++;
//            }
//            for (int j = 2; j * i < n; j++) {
//                isPrime[j * i] = 1;
//            }
//        }
//        return result;
//    }


//    public static void main(String[] args) {
//        System.out.println(new Solution().countPrimes(10));
//    }
//
//    public int countPrimes(int n) {
//        int result = 0;
//        for (int i = 2; i < n; i++) {
//            if (i != 2 && i % 2 == 0) {
//                continue;
//            }
//            double root = Math.pow(i, 0.5);
//            boolean is_prime = true;
//            for (int j = 2; j <= root; j++) {
//
//                if (i % j == 0) {
//                    is_prime = false;
//                    break;
//                }
//            }
//            if (is_prime) {
//                result++;
//            }
//        }
//        return result;
//    }

//    public static void main(String[] args) {
//        int[] test = {3,2,3,4};
//        System.out.println(new Solution().largestPerimeter(test));
//    }
//
//
//    public int largestPerimeter(int[] A) {
//        Arrays.sort(A);
//        for (int i = A.length - 1; i >= 2; i--) {
//            if (A[i] < (A[i - 1] + A[i - 2])) {
//                return A[i] + A[i - 1] + A[i - 2];
//            }
//        }
//
//        return 0;
//
//    }

//    public static void main(String[] args) {
//        int[] A = {-1, -1};
//        int[] B = {-1, 1};
//        int[] C = {-1, 1};
//        int[] D = {-1, 1};
//        System.out.println(new Solution().fourSumCount(A, B, C, D));
//
//    }
//
//
//    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
//        HashMap<Integer, Integer> map1 = new HashMap<>();
//        HashMap<Integer, Integer> map2 = new HashMap<>();
//        int result = 0;
//
//        for (int a : A) {
//            for (int b : B) {
//                int sum = a + b;
//                if (map1.containsKey(sum)) {
//                    map1.put(sum, map1.get(sum) + 1);
//                } else {
//                    map1.put(sum, 1);
//                }
//            }
//        }
//
//        for (int c : C) {
//            for (int d : D) {
//                int sum = d + c;
//                if (map2.containsKey(sum)) {
//                    map2.put(sum, map2.get(sum) + 1);
//                } else {
//                    map2.put(sum, 1);
//                }
//            }
//        }
//
//        for (int key : map1.keySet()) {
//            int rest = -key;
//            if (map2.containsKey(rest)) {
//                result = result + map1.get(key) * map2.get(rest);
//            }
//        }
//
//        return result;
//
//    }


//    public static void main(String[] args) {
//        System.out.println(new Solution().removeDuplicateLetters("cbacdcbc"));
//    }
//
//    public String removeDuplicateLetters(String s) {
//        HashMap<Character, Integer> map = new HashMap<>();
//        HashSet<Character> deal = new HashSet<>();
//        LinkedList<Character> stack = new LinkedList<>();
//        for (int i = 0; i < s.length(); i++) {
//            if (map.containsKey(s.charAt(i))) {
//                map.put(s.charAt(i), map.get(s.charAt(i)) + 1);
//            } else {
//                map.put(s.charAt(i), 0);
//            }
//        }
//
//        for (int i = 0; i < s.length(); i++) {
//            char c = s.charAt(i);
//            if (deal.contains(c)) {
//                map.put(c, map.get(c) - 1);
//                continue;
//            }
//            if (stack.size() != 0 && stack.getLast() >= c) {
//                while (stack.size() > 0 && stack.getLast() > c && map.get(stack.getLast()) > 0) {
//                    char remove = stack.removeLast();
//                    deal.remove(remove);
//                    map.put(remove, map.get(remove) - 1);
//                }
//            }
//            stack.add(c);
//            deal.add(c);
//        }
//
//        StringBuilder result = new StringBuilder(");
//        for (Character character : stack) {
//            result.append(character);
//        }
//
//        return result.toString();
//
//
//    }

//    public static void main(String[] args) {
//        String s = "1432219";
//        System.out.println(new Solution().removeKdigits(s, 3));
//    }
//
//
//    public String removeKdigits(String num, int k) {
//        int remove = k;
//        LinkedList<Character> stack = new LinkedList<>();
//        for (int i = 0; i < num.length(); i++) {
//            while (stack.size() > 0 && k > 0 && num.charAt(i) < stack.getLast()) {
//                stack.removeLast();
//                k--;
//            }
//            stack.add(num.charAt(i));
//        }
//        StringBuilder result = new StringBuilder(");
//        int remain = num.length() - remove;
//        int i = 0;
//        boolean start = true;
//        while (remain > 0 && i < stack.size()) {
//            if (stack.get(i) == '0' && start) {
//            } else {
//                result.append(stack.get(i));
//                start = false;
//            }
//            i++;
//            remain--;
//        }
//
//        if (result.length() == 0) result.append('0');
//        return result.toString();
//
//    }


//    public String removeDuplicateLetters(String s) {
//        StringBuilder result = new StringBuilder(");
//        Info[] chars = new Info[26];
//        for (int i = 0; i < 26; i++) {
//            chars[i] = new Info();
//        }
//        ArrayList<Character> list = new ArrayList<>();
//        for (int i = 0; i < s.length(); i++) {
//            char c = s.charAt(i);
//            Info in = chars[c - 'a'];
//            if (in.index == -1 || in.hasSmaller) {
//                list.add(c);
//                if (in.hasSmaller) {
//                    in.hasSmaller = false;
//                    if (in.index != -1) {
//                        list.set(in.index, '0');
//                    }
//                }
//                in.index = list.size() - 1;
//
//                for (int j = (c - 'a') + 1; j < chars.length; j++) {
//                    chars[j].hasSmaller = true;
//                }
//            }
//        }
//
//        for (Character character : list) {
//            if (character != '0') {
//                result.append(character);
//            }
//        }
//        return result.toString();
//    }
//
//    static class Info {
//        int index = -1;
//        boolean hasSmaller = false;
//    }


//    public static void main(String[] args) {
//        int[] test = {3, 2, 1, 100};
//        System.out.println(new Solution().maximumGap(test));
//    }
//
//    public int maximumGap(int[] nums) {
//        if (nums.length < 2) return 0;
//        if (nums.length == 2) {
//            return nums[0] - nums[1] > 0 ? nums[0] - nums[1] : nums[1] - nums[0];
//        }
//        ArrayList<Bucket> bucket = new ArrayList<>();
//        for (int i = 0; i < nums.length - 1; i++) {
//            bucket.add(new Bucket());
//        }
//        int max = Integer.MIN_VALUE;
//        int min = Integer.MAX_VALUE;
//
//        for (int num : nums) {
//            if (num > max) {
//                max = num;
//            }
//            if (num < min) {
//                min = num;
//            }
//        }
//
//        double gap = (double) (max - min) / (nums.length - 1);
//
//        for (int num : nums) {
//            int index = (int) ((num - min) / gap);
//            if (num == max) index = nums.length - 2;
//            Bucket b = bucket.get(index);
//            if (num > b.max) b.max = num;
//            if (num < b.min) b.min = num;
//            b.size++;
//        }
//
//        int result = 0;
//        int up = bucket.get(0).max;
//        for (int i = 1; i < bucket.size(); i++) {
//            if (bucket.get(i).size == 0) {
//                continue;
//            }
//            int dis = bucket.get(i).min - up;
//            if (dis > result) result = dis;
//            up = bucket.get(i).max;
//
//        }
//
//        return result;
//
//    }
//
//    class Bucket {
//        int size = 0;
//        int max = Integer.MIN_VALUE;
//        int min = Integer.MAX_VALUE;
//    }


//    public List<Integer> inorderTraversal(TreeNode root) {
//        List<Integer> result = new LinkedList<>();
//        LinkedList<TreeNode> stack = new LinkedList<>();
//        TreeNode p = root;
//        while (p != null || stack.size() > 0) {
//            while (p != null) {
//                stack.push(p);
//                p = p.left;
//            }
//            p = stack.pop();
//            result.add(p.val);
//            p = p.right;
//        }
//        return result;
//    }

//    public char findTheDifference(String s, String t) {
//        int[] array = new int[26];
//        for (int i = 0; i < s.length(); i++) {
//            array[s.charAt(i) - 'a']++;
//        }
//        for (int i = 0; i < t.length(); i++) {
//            array[t.charAt(i) - 'a']--;
//        }
//        for (int i = 0; i < 26; i++) {
//            if (array[i] == -1)
//                return (char) ('a' + i);
//        }
//        return 'a';
//    }

//    public String sortString(String s) {
//        int[] chars = new int[26];
//        for (int i = 0; i < s.length(); i++) {
//            chars[s.charAt(i) - 'a']++;
//        }
//        boolean forward = true;
//        int count = 0;
//        StringBuilder result = new StringBuilder(");
//        int i = 0;
//        while (count < s.length()) {
//            if (chars[i] != 0) {
//                result.append((char) ('a' + i));
//                chars[i]--;
//                count++;
//            }
//
//            if (forward) {
//                if (i == 25) {
//                    forward = false;
//                } else {
//                    i++;
//                }
//            } else {
//                if (i == 0) {
//                    forward = true;
//                } else {
//                    i--;
//                }
//
//            }
//        }
//
//        return result.toString();
//    }

//    public static void main(String[] args) {
//        System.out.println(new Solution().sortString("abbacc"));
//    }
//
//
//    public String sortString(String s) {
//        HashMap<Character, Integer> map = new HashMap<>();
//        ArrayList<Character> list = new ArrayList<>();
//        for (int i = 0; i < s.length(); i++) {
//            if (map.containsKey(s.charAt(i))) {
//                map.put(s.charAt(i), map.get(s.charAt(i)) + 1);
//            } else {
//                map.put(s.charAt(i), 1);
//                list.add(s.charAt(i));
//            }
//        }
//        Collections.sort(list);
//        boolean forward = true;
//        int count = 0;
//        int i = 0;
//        StringBuilder result = new StringBuilder(");
//        while (count < s.length()) {
//            if (forward) {
//                if (i == list.size() - 1) {
//                    if (map.get(list.get(i)) == 0) {
//                        forward = false;
//                        i--;
//                    } else {
//                        map.put(list.get(i), map.get(list.get(i)) - 1);
//                        forward = false;
//                        result.append(list.get(i));
//                        count++;
//                    }
//                } else {
//                    if (map.get(list.get(i)) == 0) {
//                        i++;
//                    } else {
//                        map.put(list.get(i), map.get(list.get(i)) - 1);
//                        result.append(list.get(i));
//                        i++;
//                        count++;
//                    }
//                }
//            } else {
//                if (i == 0) {
//                    if (map.get(list.get(i)) == 0) {
//                        forward = true;
//                        i++;
//                    } else {
//                        map.put(list.get(i), map.get(list.get(i)) - 1);
//                        forward = true;
//                        result.append(list.get(i));
//                        count++;
//                    }
//                } else {
//                    if (map.get(list.get(i)) == 0) {
//                        i--;
//                    } else {
//                        map.put(list.get(i), map.get(list.get(i)) - 1);
//                        result.append(list.get(i));
//                        i--;
//                        count++;
//                    }
//                }
//
//            }
//        }
//        return result.toString();
//    }


//    public int countNodes(TreeNode root) {
//        int result = 0;
//        if (root == null) return 0;
//        int height = 0;
//        TreeNode p = root;
//        while (p != null) {
//            height++;
//            p = p.right;
//        }
//
//        result = result + (int) Math.pow(2, height) - 1;
//        p = root;
//        for (int i = 0; i < height; i++) {
//            boolean judge = containsHalf(p, height - i);
//            if (judge) {
//                p = p.right;
//                result += Math.pow(2, height) * Math.pow(0.5, i + 1);
//            } else {
//                p = p.left;
//            }
//        }
//
//        return result;
//
//    }
//
//    public boolean containsHalf(TreeNode root, int count) {
//        TreeNode p = root;
//        p = p.left;
//        count--;
//        while (count > 0) {
//            p = p.right;
//            count--;
//        }
//        return p != null;
//    }


//    public static void main(String[] args) {
//        int[][] test = {{3, 9}, {7, 12}, {3, 8}, {6, 8}, {9, 10}, {2, 9}, {0, 9}, {3, 9}, {0, 6}, {2, 8}};
//        System.out.println(new Solution().findMinArrowShots(test));
//    }
//
//    public int findMinArrowShots(int[][] points) {
//        if (points.length == 0) {
//            return 0;
//        }
//        Interval[] balloons = new Interval[points.length];
//        for (int i = 0; i < points.length; i++) {
//            balloons[i] = new Interval(points[i][0], points[i][1]);
//        }
//        Arrays.sort(balloons);
//        int count = 1;
//        long right = balloons[0].right;
//        for (int i = 1; i < balloons.length; i++) {
//            if (balloons[i].left <= right) {
//                if(balloons[i].right<right){
//                    right=balloons[i].right;
//                }
//            } else {
//                count++;
//                right = balloons[i].right;
//            }
//        }
//        return count;
//    }
//
//    class Interval implements Comparable<Interval> {
//        long left;
//        long right;
//
//        Interval(int left, int right) {
//            this.left = left;
//            this.right = right;
//        }
//
//        @Override
//        public int compareTo(Interval o) {
//            if (Long.compare(this.left, o.left) == 0) {
//                return Long.compare(this.right, o.right);
//            } else {
//                return Long.compare(this.left, o.left);
//            }
//        }
//    }

//    public ListNode sortList(ListNode head) {
//        if (head == null || head.next == null) return head;
//        int length = 0;
//        ListNode p = head;
//        while (p != null) {
//            length++;
//            p = p.next;
//        }
//        return mergeSort(head, length);
//    }
//
//    public ListNode mergeSort(ListNode head, int length) {
//        if (length > 1) {
//            int left = length / 2;
//            int right = length - left;
//            ListNode p = head;
//            for (int i = 0; i < left - 1; i++) {
//                p = p.next;
//            }
//            ListNode p1 = p.next;
//            p.next = null;
//            ListNode leftNode = mergeSort(head, left);
//            ListNode rightNode = mergeSort(p1, right);
//            return merge(leftNode, rightNode);
//
//        } else {
//            return head;
//        }
//    }
//
//    public ListNode merge(ListNode left, ListNode right) {
//        ListNode head;
//        if (left.val < right.val) {
//            head = left;
//            left = left.next;
//        } else {
//            head = right;
//            right = right.next;
//        }
//        ListNode p = head;
//        while (left != null || right != null) {
//            if (left != null && (right == null || left.val < right.val)) {
//                p.next = left;
//                p = p.next;
//                left = left.next;
//            }
//            if (right != null && (left == null || left.val >= right.val)) {
//                p.next = right;
//                p = p.next;
//                right = right.next;
//            }
//        }
//        return head;
//    }


//    public static void main(String[] args) {
//        System.out.println(new Solution().isAnagram("aacc","ccac"));
//
//    }

//    public boolean isAnagram(String s, String t) {
//        if (s.length() != t.length()) return false;
//        HashMap<Character, Integer> map = new HashMap<>();
//        for (int i = 0; i < s.length(); i++) {
//            if (map.containsKey(s.charAt(i))) {
//                map.put(s.charAt(i), map.get(s.charAt(i)) + 1);
//            } else {
//                map.put(s.charAt(i), 1);
//            }
//        }
//        for (int i = 0; i < t.length(); i++) {
//            if (!map.containsKey(t.charAt(i)) || map.get(t.charAt(i)) <= 0) {
//                return false;
//            } else {
//                map.put(t.charAt(i), map.get(t.charAt(i)) - 1);
//            }
//        }
//
//        return true;
//    }


//    public ListNode insertionSortList(ListNode head) {
//        ListNode fakeNode = new ListNode(1);
//        fakeNode.next = head;
//        ListNode prev = fakeNode;
//        ListNode p = head;
//        while (p != null) {
//            int val = p.val;
//            ListNode p1 = fakeNode;
//            if (p.val > prev.val) {
//                prev = p;
//                p = p.next;
//                continue;
//            }
//
//
//            while (p1 != p) {
//                if (p1.next.val > val) break;
//                p1 = p1.next;
//            }
//            if (p1 == p) {
//                prev = p;
//                p = p.next;
//                continue;
//            }
//
//            ListNode p2 = p;
//            p = p.next;
//            prev.next = p;
//
//            p2.next = p1.next;
//            p1.next = p2;
//        }
//
//        return fakeNode.next;
//    }


//    public void moveZeroes(int[] nums) {
//
//        int count = 0;
//        int i = 0;
//
//        while (i < nums.length) {
//            if (nums[i] == 0) {
//                count++;
//            } else {
//                if(count == 0)
//                    continue;
//                nums[i - count] = nums[i];
//                nums[i] = 0;
//            }
//            i++;
//        }
//
//    }

//    public static void main(String[] args) {
//        int[] test1 = {1, 2, 3, 4, 5};
//        int[] test2 = {3, 4, 5, 1, 2};
//        System.out.println(new Solution().canCompleteCircuit(test1, test2));
//    }


//    public int canCompleteCircuit(int[] gas, int[] cost) {
//        for (int i = 0; i < gas.length; i++) {
//            if (gas[i] < cost[i]) {
//                continue;
//            }
//            int p, remain;
//            if (i != gas.length - 1) {
//                p = i + 1;
//            } else {
//                p = 0;
//            }
//            remain = gas[i] - cost[i];
//
//            while (p != i) {
//                if (remain < 0) break;
//                remain = remain + gas[p] - cost[p];
//                if (p == gas.length - 1) {
//                    p = 0;
//                    continue;
//                }
//                p++;
//            }
//
//            if (remain >= 0) return i;
//        }
//
//        return -1;
//    }

//
//    public int canCompleteCircuit(int[] gas, int[] cost) {
//        int i = 0;
//        while (i < gas.length) {
//            if (gas[i] < cost[i]) {
//                i++;
//                continue;
//            }
//
//            int p, remain;
//            if (i != gas.length - 1) {
//                p = i + 1;
//            } else {
//                p = 0;
//            }
//            remain = gas[i] - cost[i];
//
//            while (p != i) {
//                if (remain < 0) break;
//                remain = remain + gas[p] - cost[p];
//                if (p == gas.length - 1) {
//                    p = 0;
//                    continue;
//                }
//                p++;
//            }
//
//            if (remain >= 0) return i;
//            else if (p > i) {
//                i = p;
//            } else {
//                return -1;
//            }
//
//        }
//
//        return -1;
//    }


//    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
//        HashMap<Integer, LinkedList<Coordinate>> map = new HashMap<>();
//        int maxDis = 0;
//        int[][] result = new int[R * C][2];
//        int count = 0;
//
//
//        for (int i = 0; i < R; i++) {
//            for (int j = 0; j < C; j++) {
//                int dis = Math.abs(i - r0) + Math.abs(j - c0);
//                if (dis > maxDis) maxDis = dis;
//                if (map.containsKey(dis)) {
//                    map.get(dis).add(new Coordinate(i, j));
//                } else {
//                    Coordinate c = new Coordinate(i, j);
//                    LinkedList<Coordinate> l = new LinkedList<>();
//                    l.add(c);
//                    map.put(dis, l);
//                }
//            }
//        }
//
//        for (int i = 0; i <= maxDis; i++) {
//            LinkedList<Coordinate> l = map.get(i);
//            for (int j = 0; j < l.size(); j++) {
//                result[count][0] = l.get(j).x;
//                result[count][1] = l.get(j).y;
//                count++;
//            }
//        }
//
//        return result;
//
//
//    }
//
//    class Coordinate {
//        int x;
//        int y;
//
//        Coordinate(int x, int y) {
//            this.x = x;
//            this.y = y;
//        }
//    }
//
//
//    public static void main(String[] args) {
//
//        int[][] result = new Solution().allCellsDistOrder(3, 3, 0, 2);
//        for (int i = 0; i < result.length; i++) {
//            System.out.println(result[i][0] + " " + result[i][1]);
//        }
//
//    }
//
//    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
//        int[][] result = new int[R * C][2];
//        int count = 1;
//        LinkedList<Coordinate> stack1 = new LinkedList<>();
//        LinkedList<Coordinate> stack2 = new LinkedList<>();
//        HashSet<Coordinate> set = new HashSet<>();
//        stack1.add(new Coordinate(r0, c0));
//        set.add(new Coordinate(r0, c0));
//        result[0][0] = r0;
//        result[0][1] = c0;
//
//        while (stack1.size() > 0 || stack2.size() > 0) {
//            if (stack1.size() > 0) {
//                while(stack1.size()>0) {
//                    Coordinate c = stack1.pop();
//                    if (c.x > 0) {
//                        Coordinate c1 = new Coordinate(c.x - 1, c.y);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack2.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.y > 0) {
//                        Coordinate c1 = new Coordinate(c.x, c.y - 1);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack2.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.x < R - 1) {
//                        Coordinate c1 = new Coordinate(c.x + 1, c.y);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack2.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.y < C - 1) {
//                        Coordinate c1 = new Coordinate(c.x, c.y + 1);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack2.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//
//                }
//            } else {
//                while (stack2.size() > 0) {
//                    Coordinate c = stack2.pop();
//                    if (c.x > 0) {
//                        Coordinate c1 = new Coordinate(c.x - 1, c.y);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack1.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.y > 0) {
//                        Coordinate c1 = new Coordinate(c.x, c.y - 1);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack1.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.x < R - 1) {
//                        Coordinate c1 = new Coordinate(c.x + 1, c.y);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack1.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//                    if (c.y < C - 1) {
//                        Coordinate c1 = new Coordinate(c.x, c.y + 1);
//                        if (!set.contains(c1)) {
//                            set.add(c1);
//                            stack1.push(c1);
//                            result[count][0] = c1.x;
//                            result[count][1] = c1.y;
//                            count++;
//                        }
//                    }
//
//
//                }
//
//            }
//        }
//
//
//        return result;
//    }
//
//    static class Coordinate {
//
//        int x;
//        int y;
//
//        Coordinate(int x, int y) {
//            this.x = x;
//            this.y = y;
//        }
//
//        @Override
//        public boolean equals(Object o) {
//            if (this == o) return true;
//            if (o == null || getClass() != o.getClass()) return false;
//            Coordinate that = (Coordinate) o;
//            return x == that.x &&
//                    y == that.y;
//        }
//
//        @Override
//        public int hashCode() {
//            return Objects.hash(x, y);
//        }
//    }


}


class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}


class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node parent;
}
