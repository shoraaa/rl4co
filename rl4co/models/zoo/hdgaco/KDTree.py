import torch
import math

class Node:
    __slots__ = ('is_leaf', 'cut_dim', 'cut_val', 'left', 'right', 'indices')
    def __init__(self):
        self.is_leaf = False
        self.cut_dim = -1
        self.cut_val = 0.0
        self.left = None
        self.right = None
        self.indices = None

class KDTree:
    def __init__(self, points, leaf_size=16):
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float)
        self.points = points
        self.leaf_size = leaf_size
        self.n, self.dim = points.shape
        self.device = points.device
        self.root = self._build_tree(0, self.n - 1)

    def _build_tree(self, low, high):
        n = high - low + 1
        if n <= self.leaf_size:
            node = Node()
            node.is_leaf = True
            node.indices = torch.arange(low, high + 1, device=self.device)
            return node
        
        pts = self.points[low:high + 1]
        min_vals, _ = torch.min(pts, dim=0)
        max_vals, _ = torch.max(pts, dim=0)
        spreads = max_vals - min_vals
        cut_dim = torch.argmax(spreads).item()
        
        vals = self.points[low:high + 1, cut_dim]
        sorted_idx = torch.argsort(vals)
        sorted_segment = torch.arange(low, high + 1, device=self.device)[sorted_idx]
        self.points[low:high + 1] = self.points[sorted_segment]
        
        m = low + (high - low) // 2
        median_val = self.points[m, cut_dim].item()
        
        left_node = self._build_tree(low, m)
        right_node = self._build_tree(m + 1, high)
        
        node = Node()
        node.is_leaf = False
        node.cut_dim = cut_dim
        node.cut_val = median_val
        node.left = left_node
        node.right = right_node
        return node

    def query(self, query_point, k=1):
        if not isinstance(query_point, torch.Tensor):
            query_point = torch.tensor(query_point, dtype=torch.float, device=self.device)
        best_dist = torch.tensor(float('inf'), device=self.device)
        best_index = torch.tensor(-1, dtype=torch.long, device=self.device)
        best_dist, best_index = self._query(self.root, query_point, best_dist, best_index)
        return best_dist.item(), best_index.item()

    def _query(self, node, query_point, best_dist, best_index):
        if node.is_leaf:
            pts = self.points[node.indices]
            dists = torch.norm(pts - query_point, dim=1)
            min_val, min_idx = torch.min(dists, dim=0)
            if min_val < best_dist:
                best_dist = min_val
                best_index = node.indices[min_idx]
            return best_dist, best_index
        
        coord_val = query_point[node.cut_dim].item()
        if coord_val <= node.cut_val:
            best_dist, best_index = self._query(node.left, query_point, best_dist, best_index)
            if coord_val + best_dist > node.cut_val:
                best_dist, best_index = self._query(node.right, query_point, best_dist, best_index)
        else:
            best_dist, best_index = self._query(node.right, query_point, best_dist, best_index)
            if coord_val - best_dist < node.cut_val:
                best_dist, best_index = self._query(node.left, query_point, best_dist, best_index)
        return best_dist, best_index

    @staticmethod
    def brute_force(points, query_point):
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float)
        if not isinstance(query_point, torch.Tensor):
            query_point = torch.tensor(query_point, dtype=torch.float)
        dists = torch.norm(points - query_point, dim=1)
        min_val, min_idx = torch.min(dists, dim=0)
        return min_val.item(), min_idx.item()


if __name__ == "__main__":
    # Create some random data:
    torch.manual_seed(0)
    pts = torch.randn(10000, 2)        # 1000 2D points
    tree = KDTree(pts)

    k = 84

    # for i in range(len(pts)):
    #     q = pts[i]
    #     # Query the KD-Tree:
    #     knn = tree.query(q, k)
