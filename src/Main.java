import java.util.*;

public class Main {
  public static void main(String[] args) {
    int[][] arr2d = {{10, 21, 23}, {43, 45, 65}, {79, 87, 69}, {82, 50, 70, 76, 56, 14}};
    int[] arr = {10, 2, 234, 45, 657, 87, 698, 500, 7656, 1};
    System.out.println(findNumbers(arr));
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

  public static int recursiveFibonacci(int n) {
    if (n <= 1) return n;
    return recursiveFibonacci(n - 1) + recursiveFibonacci(n - 2);
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

  public static int countDigits(int n, int target) {
    int count = 0;
    while (n > 0) {
      int rem = n % 10;
      if (rem == target) count++;
      n /= 10;
    }
    return count;
  }

  public static int reverseDigits(int n) {
    int reverse = 0;
    while (n > 0) {
      reverse = reverse * 10 + n % 10;
      n /= 10;
    }
    return reverse;
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

  public static void swap(int[] arr, int index1, int index2) {
    int temp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = temp;
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

  public static int linearSearchArray(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++)
      if (arr[i] == target) return i;
    return -1;
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
   * <a href="https://leetcode.com/problems/find-numbers-with-even-number-of-digits/">
   *   1295. Find Numbers with Even Number of Digits
   * </a>
   */
  public static int findNumbers(int[] nums) {
    if(nums.length == 0) return 0;
    int count = 0;
    for (int num : nums)
      if (((int)(Math.log10(num)) & 1) == 1) count++;
    return count;
  }

  public static boolean isEven(int n) {
    return countDigitOptimized(n) % 2 == 0;
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
   *   ...
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
   *   ...
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
   * <a href="https://leetcode.com/problems/">
   *   ...
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

  /**
   * <a href="https://leetcode.com/problems/">
   *   ...
   * </a>
   */
  public static boolean containsDuplicate(int[] nums) {
    if (nums.length == 1) return false;
    Set<Integer> set = new HashSet<>();
    for (int num : nums)
      if (!set.add(num)) return true;
    return false;
  }

  /**
   * <a href="https://leetcode.com/problems/">
   *   ...
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
   * <a href="https://leetcode.com/problems/">
   *   ...
   * </a>
   */
  public static String longestCommonPrefix(String[] strs) {
    return "";
  }
}

//Input: strs = ["flower","flow","flight"]
//Output: "fl"
//Example 2:
//
//Input: strs = ["dog","racecar","car"]
//Output: ""
//Explanation: There is no common prefix among the input strings.
