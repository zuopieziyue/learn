"""
主要是热题HOT100的题目
"""

import os
import Optional
import collections


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






class Solution:
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)

            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain

            # 更新答案
            self.maxSum = max(self.maxSum, priceNewpath)

            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)

        maxGain(root)
        return self.maxSum




