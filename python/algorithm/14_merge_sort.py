# coding: utf-8

def merge_sort(alist):
    """归并排序"""
    n = len(alist)
    if n <= 1:
        return
    mid = n // 2
    # left 采用归并排序后形成的有序的新的列表
    left = merge_sort(alist[:mid])

    # right 采用归并排序后形成的有序的新的列表
    right = merge_sort(alist[mid:])

    # 将两个有序的子序列合并为一个新的整体
    # merge(left, right)
    left_pointer, right_pointer = 0, 0




