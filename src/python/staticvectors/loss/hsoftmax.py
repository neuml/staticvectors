"""
HierarchicalSoftmaxLoss module
"""

import heapq

import numpy as np

from .base import Loss


class HierarchicalSoftmaxLoss(Loss):
    """
    Hierarchical softmax loss. This method estimates softmax with Huffman coding.

    See the following links for more on this algorithm.
      - https://arxiv.org/abs/1607.01759
      - https://github.com/facebookresearch/fastText/blob/main/src/loss.cc#L192
    """

    def __init__(self, counts, weights):
        # Call parent constructor
        super().__init__(counts, weights)

        # Build the Huffman tree
        self.root = self.build(counts)

    def __call__(self, vector, limit):
        """
        Predicts the label for vector

        Args:
            vector: input vector
            limit: max labels to return

        Returns:
            [(label id, score)]
        """

        # Collect scores with depth first search
        scores = self.dfs(self.root, vector, limit)
        return [(uid, np.exp(score)) for score, uid in scores]

    def build(self, counts):
        """
        Builds a Huffman tree.

        Args:
            counts: label frequency count

        Returns:
            root node
        """

        index, heap = 0, []
        for uid, count in counts.items():
            node = Node(uid, count)
            heapq.heappush(heap, node)
            index += 1

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            # Merge node
            merged = Node(index, left.count + right.count)

            # Leaf nodes
            merged.left = left
            merged.right = right

            # Save node and push on heap
            index += 1
            heapq.heappush(heap, merged)

        # Return root node
        return heapq.heappop(heap)

    def dfs(self, node, vector, limit, scores=None, score=0.0):
        """
        Depth first search algorithm.

        Args:
            node: starting node
            vector: input vector
            limit: maximum number of labels to return
            scores: list of scores for this search
            score: current score for this search

        Returns:
            list of [score, node id]
        """

        # Initialize scores
        scores = scores if scores is not None else []

        # Stop traversing tree when current score is lower than current lowest score
        if len(scores) == limit and score < scores[0][0]:
            return scores

        # Collect scores
        if node.left is None and node.right is None:
            heapq.heappush(scores, (score, node.id))
            if len(scores) > limit:
                heapq.heappop(scores)

            return scores

        # Calculate softmax for this node
        f = np.dot(vector, self.weights[node.id - self.weights.shape[0]])
        f = 1.0 / (1 + np.exp(-f))

        # Traverse tree nodes
        self.dfs(node.left, vector, limit, scores, score + np.log((1.0 - f) + 1e-5))
        self.dfs(node.right, vector, limit, scores, score + np.log(f + 1e-5))

        return scores


class Node:
    """
    Tree node.
    """

    def __init__(self, uid, count):
        """
        Creates a tree node.

        Args:
            uid: node id
            count: node frequency count
        """

        self.id = uid
        self.count = count
        self.left = None
        self.right = None

    def __lt__(self, other):
        """
        Node comparator.
        """

        return self.count < other.count
