"""
主要是热题HOT100的题目
"""

import os
import Optional
import collections
import functools
import math


# 定义二叉树节点
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution_96:
    """不同的二叉搜索树"""
    def numTrees(self,n):
        G = [0]*(n+1)
        G[0], G[1] = 1, 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]
        return G[n]


class Solution_98:
    """验证二叉搜索树"""
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True

        return helper(root)


class Solution_101:
    """对称二叉树"""
    def isSymmetric(self, root)
        def check(p, q):
            if p is None and q is None:
                return True
            if p is None or q is None:
                return False
            return p.val==q.val and check(p.left, q.right) and check(p.right, q.left)

        return check(root, root)


class Solution_102:
    """二叉树的层序遍历，广度优先搜索"""
    def levelOrder(self, root):
        if root == None:
            return []
        que = collections.deque([root])
        ans = []
        while len(que) != 0:
            size = len(que)
            level = []
            for _ in range(size):
                cur = que.popleft()
                level.append(cur.val)
                if cur.left != None:
                    que.append(cur.left)
                if cur.right != None:
                    que.append(cur.right)
            ans.append(level)
        return ans


class Solution_103:
    """二叉树的锯齿形层序遍历， 广度有限搜索"""
    def zigzagLevelOrder(self, root):
        if root == None:
            return []
        que = collections.deque([root])
        ans = []
        is_order_left = True

        while len(que) != 0:
            size = len(que)
            level = []
            for _ in range(size):
                cur = que.popleft()
                level.append(cur.val)
                if cur.left != None:
                    que.append(cur.left)
                if cur.right != None:
                    que.append(cur.right)
            if not is_order_left:
                level.reverse()
            ans.append(level)
            is_order_left = not is_order_left

        return ans


class Solution_104:
    """二叉树的最大深度，深度优先搜索"""
    def maxDepth(self, root):
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
            return max(left_height, right_height) + 1


class Solution_105:
    """从前序和中序遍历序列构造二叉树"""
    def buildTree(self, preorder, inorder):
        if not preorder:
            return None

        root = TreeNode(preorder[0])
        stack = [root]
        inorderIndex = 0
        for i in range(1, len(preorder)):
            preorderVal = preorder[i]
            node = stack[-1]
            if node.val != inorder[inorderIndex]:
                node.left = TreeNode(preorderVal)
                stack.append(node.left)
            else:
                while stack and stack[-1].val == inorder[inorderIndex]:
                    node = stack.pop()
                    inorderIndex += 1
                node.right = TreeNode(preorderVal)
                stack.append(node.right)

        return root


class Solution_114:
    """二叉树展开为链表，即二叉树的前序遍历（递归方式）"""
    def flatten(self, root):
        preorderList = list()

        def preorderTraversal(root):
            if root:
                preorderList.append(root)
                preorderTraversal(root.left)
                preorderTraversal(root.right)

        preorderTraversal(root)
        my_size = len(preorderList)
        for i in range(1, my_size):
            prev, curr = preorderList[i-1], preorderList[i]
            prev.left = None
            prev.right = curr


class Solution_121:
    """买卖股票的最佳时机"""
    def maxProfit(self, prices) :
        inf = int(1e9)
        minprice = inf
        maxprofit = 0
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)
            minprice = min(price, minprice)
        return maxprofit


class Solution_124:
    """二叉树中的最大路径和"""
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root):
        def maxGain(node):
            if not node:
                return 0
            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)

            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewPath = node.val + leftGain + rightGain

            # 更新答案
            self.maxSum = max(self.maxSum, priceNewPath)

            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)

        maxGain(root)
        return self.maxSum


class Solution_128(object):
    """最长连续序列"""
    def longestConsecutive(self, nums):
        hash_dict = dict()

        max_length = 0
        for num in nums:
            if num not in hash_dict:
                left = hash_dict.get(num-1, 0)
                right = hash_dict.get(num+1, 0)

                cur_length = 1 + left + right
                if cur_length > max_length:
                    max_length = cur_length

                hash_dict[num] = cur_length
                hash_dict[num - left] = cur_length
                hash_dict[num + right] = cur_length

        return max_length


class Solution_136:
    """只出现一次的数字"""
    def singleNumber(self, nums):
        return functools.reduce(lambda x, y: x ^ y, nums)


class Solution_139:
    """单词拆分"""
    def wordBreak(self, s, wordDict):
        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True
        for i in range(n):
            for j in range(i+1, n+1):
                if(dp[i] and (s[i:j] in wordDict)):
                    dp[j] = True
        return dp[-1]


class Solution_141:
    """环形链表"""
    def hasCycle(self, head):
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False


class Solution_142:
    """环形链表II"""
    def hasCycle(self, head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next):
                return
            fast, slow = fast.next.next, slow.next
            if fast == slow:
                break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast


class DLinkedNode:
    """双向链表"""
    def __init__(self,key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache_146:
    """LRU缓存机制"""

    def __init__(self, capacity):
        self.cache = dict()
        # 使用伪头部和伪尾部节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self,key):
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key, value):
        if key not in self.chche:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node


class Solution_138:
    """排序链表"""
    def sortList(self, head):
        def sortFunc(head, tail):
            if not head:
                return head
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))

        def merge(head1, head2):
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next

        return sortFunc(head, None)


class Solution_152:
    """成积最大子数组"""
    """
        我们只要记录前 i 的最小值和最大值，
        那么 dp[i] = max(nums[i] * pre_max, nums[i] * pre_min, nums[i])，
        这里 0 不需要单独考虑，因为当相乘不管最大值和最小值，都会置 0
    """
    def maxProduct(self, nums):
        if not nums:
            return
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]
        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min

        return res


class Solution_155:
    """最小栈"""
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]

    def push(self, x):
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]


