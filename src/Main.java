import java.util.*;

public class Main {
  public static void main(String[] args) {
    int[][] arr2d = {{10, 21, 23}, {43, 45, 65}, {79, 87, 69}, {82, 50, 70, 76, 56, 14}};
    int[] arr = {10, 2, 234, 45, 657, 87, 698, 500, 7656, 1};
    int[] sortedArrAsc = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    int[] sortedArrDesc = {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int[] nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int[] nums1 = {1, 2, 5};
    int[] nums2 = {2};
    var left = 11;
    var right = 4;
    var text1 = "abcde";
    var text2 = "ace";
    System.out.println(coinChange(nums1, left));
  }

  public static void func1() {
    try {
      System.out.println("call func 2");
      String res = func2();
      System.out.println(res);
    } catch (Exception e) {
      System.out.println("helping 2 handle");
      System.out.println(e.getMessage());
    }
  }

  public static String func2() throws Exception {
    System.out.println("second except");
    throw new Exception("Thoughts from 1");
  }

  public static int[] array2DSearch(int[][] arr, int target) {
    if (arr.length == 0) return new int[]{-1, -1};
    for (int row = 0; row < arr.length; row++) {
      for (int col = 0; col < arr[row].length; col++) {
        if (arr[row][col] == target) return new int[]{row, col};
      }
    }
    return new int[]{-1, -1};
  }

  public static int array2DMax(int[][] arr) {
    int max = Integer.MIN_VALUE;
    for (int[] row : arr)
      for (int col : row)
        if (col > max) max = col;
    return max;
  }

  public static int array2DMin(int[][] arr) {
    int min = Integer.MAX_VALUE;
    for (int[] row : arr)
      for (int col : row)
        if (col < min) min = col;
    return min;
  }

  /**
   * <a href="https://leetcode.com/problems/coin-change/description/">
   * You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
   * Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
   * You may assume that you have an infinite number of each kind of coin.
   * </a>
   */
  public static int coinChange(int[] coins, int amount) {
    if (amount < 0) return -1;
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, Integer.MAX_VALUE);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
      for (int coin : coins)
        if (i - coin >= 0 && dp[i - coin] != Integer.MAX_VALUE)
          dp[i] = Math.min(dp[i], dp[i - coin] + 1);
    return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
  }

  /**
   * <a href="https://leetcode.com/problems/coin-change-11/description/">
   * You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
   * Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
   * You may assume that you have an infinite number of each kind of coin.
   * The answer is guaranteed to fit into a signed 32-bit integer.
   * </a>
   */
  public static int change(int amount, int[] coins) {
    return 0;
  }

  public static int countDigits(int n, int target) {
    int count = 0;
    while (n > 0) {
      int rem = n % 10;
      if (rem == target) count++;
      n /= 10;
    }
    return count;
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static boolean containsDuplicate(int[] nums) {
    if (nums.length == 1) return false;
    Set<Integer> set = new HashSet<>();
    for (int num : nums)
      if (!set.add(num)) return true;
    return false;
  }

  public static int countDigitOptimized(int n) {
    if (n < 0) n *= -1;
    if (n == 0) return 1;
    return (int) (Math.log10(n) + 1);
  }

  public static int countDigit(int n) {
    if (n < 0) n *= -1;
    if (n == 0) return 1;
    int count = 0;
    while (n > 0) {
      count++;
      n /= 10;
    }
    return count;
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static char findTheDifference(String s, String t) {
    int x = 0;
    for (char i : s.toCharArray())
      x ^= i;
    for (char i : t.toCharArray())
      x ^= i;
    return (char) x;
  }

  /**
   * <a href="https://leetcode.com/problems/find-numbers-with-even-number-of-digits/">
   * Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
   * The overall run time complexity should be O(log (m+n)).
   * </a>
   */
  public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int n1 = nums1.length, n2 = nums2.length;
    //if n1 is bigger swap the arrays:
    if (n1 > n2) return findMedianSortedArrays(nums2, nums1);
    int n = n1 + n2; //total length
    int left = (n + 1) / 2; //length of left half
    //apply binary search:
    int low = 0, high = n1;
    while (low <= high) {
      int mid1 = (low + high) / 2;
      int mid2 = left - mid1;
      //calculate l1, l2, r1 and r2;
      int l1 = (mid1 > 0) ? nums1[mid1 - 1] : Integer.MIN_VALUE;
      int l2 = (mid2 > 0) ? nums2[mid2 - 1] : Integer.MIN_VALUE;
      int r1 = (mid1 < n1) ? nums1[mid1] : Integer.MAX_VALUE;
      int r2 = (mid2 < n2) ? nums2[mid2] : Integer.MAX_VALUE;
      if (l1 <= r2 && l2 <= r1) {
        if (n % 2 == 1) return Math.max(l1, l2);
        else return ((double) (Math.max(l1, l2) + Math.min(r1, r2))) / 2.0;
      } else if (l1 > r2) high = mid1 - 1;
      else low = mid1 + 1;
    }
    return 0;
  }

  /**
   * <a href="https://leetcode.com/problems/find-numbers-with-even-number-of-digits/">
   * 1295. Find Numbers with Even Number of Digits
   * </a>
   */
  public static int findNumbers(int[] nums) {
    if (nums.length == 0) return 0;
    int count = 0;
    for (int num : nums)
      if (((int) (Math.log10(num)) & 1) == 1) count++;
    return count;
  }

  public static boolean isEven(int n) {
    return countDigitOptimized(n) % 2 == 0;
  }

  public static int iterativeBinarySearch(int[] arr, int target) {
    int start = 0;
    int end = arr.length - 1;
    while (start <= end) {
      int mid = start + (end - start) / 2;
      if (target < arr[mid]) end = mid - 1;
      else if (target > arr[mid]) start = mid + 1;
      else return mid;
    }
    return -1;
  }

  public static int iterativeFibonacci(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    int prev = 0;
    int cur = 0;
    for (int i = 1; i <= n; i++) {
      int next = prev + cur;
      cur = prev;
      prev = next;
    }
    return cur;
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static boolean isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) return false;
    int lastDigit = 0;
    while (x > lastDigit) {
      lastDigit = lastDigit * 10 + x % 10;
      x /= 10;
    }
    return x == lastDigit || x == lastDigit / 10;
  }

  /**
   * <a href="https://leetcode.com/problems/kth-largest-element-in-an-array/description/">
   * Given an integer array nums and an integer k, return the kth largest element in the array.
   * Note that it is the kth largest element in the sorted order, not the kth distinct element.
   * Can you solve it without sorting?
   * </a>
   */
  public static int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    for (int num : nums) {
      minHeap.add(num);
      if (minHeap.size() > k) minHeap.poll();
    }
    return minHeap.peek();
  }

  public static int linearSearchArray(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++)
      if (arr[i] == target) return i;
    return -1;
  }

  /**
   * <a href="https://leetcode.com/problems/longest-common-subsequence/description/">
   * Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
   * A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
   * For example, "ace" is a subsequence of "abcde".
   * A common subsequence of two strings is a subsequence that is common to both strings.
   * </a>
   */
  public static int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length();
    int n = text2.length();
    int[][] LCS_table = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        if (text1.charAt(i - 1) == text2.charAt(j - 1)) LCS_table[i][j] = LCS_table[i - 1][j - 1] + 1;
        else LCS_table[i][j] = Math.max(LCS_table[i - 1][j], LCS_table[i][j - 1]);
      }
    }
    return LCS_table[m][n];
  }

  public static int minArray(int[] arr) {
    int min = Integer.MAX_VALUE;
    for (int j : arr)
      if (j < min) min = j;
    return min;
  }

  public static int maxArray(int[] arr) {
    int max = Integer.MIN_VALUE;
    for (int j : arr)
      if (j > max) max = j;
    return max;
  }

  /**
   * <a href="https://leetcode.com/problems/maximum-product-subarray/description/">
   * Given an integer array nums, find a subarray that has the largest product, and return the product.
   * The test cases are generated so that the answer will fit in a 32-bit integer.
   * </a>
   */
  public static int maxProduct(int[] nums) {
    int n = nums.length;
    int maxProd = Integer.MIN_VALUE;
    int leftToRight = 1;
    int rightToLeft = 1;
    for (int i = 0; i < n; i++) {
      if (leftToRight == 0) leftToRight = 1;
      if (rightToLeft == 0) rightToLeft = 1;
      leftToRight *= nums[i];
      int j = n - i - 1;
      rightToLeft *= nums[j];
      maxProd = Math.max(leftToRight, Math.max(rightToLeft, maxProd));
    }
    return maxProd;
  }

  /**
   * <a href="https://leetcode.com/problems/richest-customer-wealth/description/">
   * 1672. Richest Customer Wealth
   * </a>
   */
  public static int maximumWealth(int[][] accounts) {
    int richest = 0;
    for (int[] account : accounts) {
      int wealth = 0;
      for (int n : account) wealth += n;
      if (wealth > richest) richest = wealth;
    }
    return richest;
  }

  /**
   * <a href="https://leetcode.com/problems/maximum-subarray/">
   * Given an integer array nums, find the subarray with the largest sum, and return its sum.
   * </a>
   */
  public static int maxSubArray(int[] nums) {
    var maxSoFar = Integer.MIN_VALUE;
    var maxEndingHere = 0;
    for (int num : nums) {
      maxEndingHere += num;
      if (maxSoFar < maxEndingHere) maxSoFar = maxEndingHere;
      if (maxEndingHere < 0) maxEndingHere = 0;
    }
    return maxSoFar;
  }

  /**
   * <a href="https://leetcode.com/problems/merge-sorted-array/description/">
   * You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
   * Merge nums1 and nums2 into a single array sorted in non-decreasing order.
   * The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
   * </a>
   */
  public static void merge(int[] nums1, int m, int[] nums2, int n) {
    int left = m - 1, right = n - 1, gap = m + n - 1;
    while (left >= 0 && right >= 0) {
      if (nums1[left] > nums2[right]) nums1[gap--] = nums1[left--];
      else nums1[gap--] = nums2[right--];
    }
    while (right >= 0) nums1[gap--] = nums2[right--];
  }

  public static int orderAgnosticIterativeBinarySearch(int[] arr, int target) {
    int start = 0;
    int end = arr.length - 1;
    boolean isAscending = arr[start] < arr[end];
    while (start <= end) {
      int mid = start + (end - start) / 2;
      if (arr[mid] == target) return mid;
      if (isAscending) {
        if (target < arr[mid]) end = mid - 1;
        else start = mid + 1;
      } else {
        if (target > arr[mid]) end = mid - 1;
        else start = mid + 1;
      }
    }
    return -1;
  }

  public static int orderAgnosticRecursiveBinarySearch(int[] arr, int target) {
    boolean isAscending = arr[0] < arr[arr.length - 1];
    return orderAgnosticRecursiveBinarySearch(arr, target, 0, arr.length - 1, isAscending);
  }

  public static int orderAgnosticRecursiveBinarySearch(int[] arr, int target, int start, int end, boolean isAscending) {
    if (start > end) return -1;
    int mid = start + (end - start) / 2;
    if (arr[mid] == target) return mid;
    if (isAscending) {
      if (target < arr[mid]) return orderAgnosticRecursiveBinarySearch(arr, target, start, mid - 1, true);
      else return orderAgnosticRecursiveBinarySearch(arr, target, mid + 1, end, true);
    } else {
      if (target > arr[mid]) return orderAgnosticRecursiveBinarySearch(arr, target, start, mid - 1, false);
      else return orderAgnosticRecursiveBinarySearch(arr, target, mid + 1, end, false);
    }
  }

  public static int recursiveBinarySearch(int[] arr, int target) {
    return recursiveBinarySearch(arr, target, 0, arr.length - 1);
  }

  public static int recursiveBinarySearch(int[] arr, int target, int start, int end) {
    if (start > end) return -1;
    int mid = start + (end - start) / 2;
    if (arr[mid] == target) return mid;
    else if (target < arr[mid]) return recursiveBinarySearch(arr, target, start, mid - 1);
    else return recursiveBinarySearch(arr, target, mid + 1, end);
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static int romanToInt(String s) {
    HashMap<Character, Integer> map = new HashMap<>();
    map.put('I', 1);
    map.put('V', 5);
    map.put('X', 10);
    map.put('L', 50);
    map.put('C', 100);
    map.put('D', 500);
    map.put('M', 1000);
    int num = 0;
    int prevValue = 0;
    for (int i = 0; i < s.length(); i++) {
      char curChar = s.charAt(i);
      int curValue = map.get(curChar);
      if (curValue > prevValue) {
        num += curValue - 2 * prevValue;
      } else {
        num += curValue;
      }
      prevValue = curValue;
    }
    return num;
  }

  public static int recursiveFibonacci(int n) {
    if (n <= 1) return n;
    return recursiveFibonacci(n - 1) + recursiveFibonacci(n - 2);
  }

  public static int[] reverseArr(int[] arr) {
    int start = 0;
    int end = arr.length - 1;
    while (start < end) {
      swap(arr, start, end);
      start++;
      end--;
    }
    return arr;
  }

  public static int reverseDigits(int n) {
    int reverse = 0;
    while (n > 0) {
      reverse = reverse * 10 + n % 10;
      n /= 10;
    }
    return reverse;
  }

  public static class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
      this.val = val;
    }

    ListNode(int val, ListNode next) {
      this.val = val;
      this.next = next;
    }
  }

  /**
   * <a href="https://leetcode.com/problems/search-in-rotated-sorted-array/description/">
   * Given the head of a singly linked list, reverse the list, and return the reversed list.
   * </a>
   */
  public static ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) return head;
//    ListNode prev, curr, next;
//    prev = null; curr = head; next = head.next;
//    while (curr != null) {
//      curr.next = prev;
//      prev = curr;
//      curr = next;
//      if (next != null) next = next.next;
//    }
//    return prev;
//    Recursion start
    var last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
//    Recursion end
  }

  /**
   * <a href="https://leetcode.com/problems/search-in-rotated-sorted-array/description/">
   * Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.
   * </a>
   */
  public static ListNode reverseBetween(ListNode head, int left, int right) {
    // If left and right are the same, no reversal is needed
    if (left == right) return head;
    ListNode revs = null, revs_prev = null;
    ListNode revend = null, revend_next = null;
    // Traverse the list to locate the nodes and pointers needed for reversal
    int i = 1;
    ListNode currNode = head;
    while (currNode != null && i <= right) {
      // Track the node just before the start of the reversal segment
      if (i < left) revs_prev = currNode;
      // Track the start of the reversal segment
      if (i == left) revs = currNode;
      // Track the end of the reversal segment and the node right after it
      if (i == right) {
        revend = currNode;
        revend_next = currNode.next;
      }
      currNode = currNode.next;
      i++;
    }
    // Detach the segment to be reversed from the rest of the list
    if (revs != null) revend.next = null;
    // Reverse the segment from position left to right
    revend = reverse(revs);
    // Reattach the reversed segment back to the list If the reversal segment was not at the head of the list
    if (revs_prev != null) revs_prev.next = revend;
    else head = revend;
    // Connect the end of the reversed segment to the rest of the list
    if (revs != null) revs.next = revend_next;
    return head;
  }

  static ListNode reverse(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
      var nextNode = curr.next;
      curr.next = prev;
      prev = curr;
      curr = nextNode;
    }
    return prev;
  }

  /**
   * <a href="https://leetcode.com/problems/search-in-rotated-sorted-array/description/">
   * There is an integer array nums sorted in ascending order (with distinct values).
   * Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
   * Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
   * You must write an algorithm with O(log n) runtime complexity.
   * </a>
   */
  public static int search(int[] nums, int target) {
    int low = 0, high = nums.length - 1;
    while (low <= high) {
      int mid = (low + high) / 2;
      if (nums[mid] == target) return mid;
      if (nums[low] <= nums[mid]) {
        if (nums[low] <= target && target <= nums[mid]) high = mid - 1;
        else low = mid + 1;
      } else {
        if (nums[mid] <= target && target <= nums[high]) low = mid + 1;
        else high = mid - 1;
      }
    }
    return -1;
  }

  public static void swap(int[] arr, int index1, int index2) {
    int temp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = temp;
  }

  public static boolean stringSearch(String s, char target) {
    if (s.isEmpty()) return false;
    for (int i = 0; i < s.length(); i++)
      if (s.charAt(i) == target) return true;
    return false;
  }

  public static int searchInRange(int[] arr, int target, int start, int end) {
    if (arr == null || arr.length == 0) return -1;
    for (int i = start; i <= end; i++)
      if (arr[i] == target) return i;
    return -1;
  }

  public static ArrayList<Integer> twoSumBruteForce(int[] arr, int target) {
    ArrayList<Integer> res = new ArrayList<>();
    for (int i = 0; i < arr.length; i++) {
      for (int j = i + 1; j < arr.length; j++) {
        if (arr[i] + arr[j] == target) {
          res.add(i);
          res.add(j);
          return res;
        }
      }
    }
    return res;
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static int[] twoSumOptimized(int[] arr, int target) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < arr.length; i++) {
      int difference = target - arr[i];
      if (map.containsKey(difference)) return new int[]{map.get(difference), i};
      map.put(arr[i], i);
    }
    return new int[0];
  }

  /**
   * <a href="https://leetcode.com/problems/">
   * ...
   * </a>
   */
  public static String longestCommonPrefix(String[] s) {
    return "";
  }
}
