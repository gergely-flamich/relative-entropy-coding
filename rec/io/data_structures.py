import numpy as np
from tqdm import tqdm


class TreeLeaf:
    """
    # ------------------------------------------------------------------------------
    # Tree Nodes
    # ------------------------------------------------------------------------------
    """
    def __init__(self, val, idx):
        # Sample value
        self.val = val

        # Sample index
        self.idx = idx

        # Children
        self.left = None
        self.right = None
        
        self.parent = None
        
        self.height = 0
        
    @property
    def left_height(self):
        return -1 if self.left is None else self.left.height
    
    @property
    def right_height(self):
        return -1 if self.right is None else self.right.height

    def __str__(self):
        return "{} : {} - {}".format(self.val, self.idx, self.height)
    

class IntervalAVLTree:
    """
    # ------------------------------------------------------------------------------
    # AVL self balancing tree
    # ------------------------------------------------------------------------------
    """

    def __init__(self, nodes, show_loading=True):

        self.root = None

        self.depth = 0

        if show_loading:
            print("Building tree")
            for i in tqdm(range(len(nodes))):
                self.add(nodes[i], i)
            
        else:
            for i in range(len(nodes)):
                self.add(nodes[i], i)

    def add(self, val, idx):

        # --------------------------------------------
        # Step 1: Usual BST insertion
        # --------------------------------------------
        
        node_list = []
        
        leaf = TreeLeaf(val, idx)

        if self.root is None:
            self.root = leaf
            self.depth = 1
        else:
            current = self.root
            depth = 1

            while True:
                node_list.append(current)
                
                depth += 1

                if val < current.val:

                    if current.left is None:
                        current.left = leaf
                        break
                    else:
                        current = current.left

                elif val > current.val:

                    if current.right is None:
                        current.right = leaf
                        break
                    else:
                        current = current.right

            # Set the parent of the node
            leaf.parent = current
            
            if self.depth < depth:
                self.depth = depth
                
        # --------------------------------------------
        # Step 2: Restore AVL property
        # --------------------------------------------
        
        # Check imbalances
        for node in node_list[::-1]:
            
            # Check if the AVL property is violated
            if abs(node.left_height - node.right_height) > 1:
                
                # Check if we're in the right-heavy case:
                if node.left_height < node.right_height:
                
                    # right child is right-heavy / balanced: right rotate
                    if node.right.left_height <= node.right.right_height:
                        self.right_rotate(node)

                    # right child is left-heavy: two rotations
                    else:
                        self.right_rotate(node.right)
                        self.left_rotate(node)
                    
                # Otherwise we're in the left-heavy case:
                else:
                    # left child is left-heavy / balanced: left rotate
                    if node.right.left_height <= node.right.right_height:
                        self.left_rotate(node)

                    # left child is right-heavy: two rotations
                    else:
                        self.left_rotate(node.left)
                        self.right_rotate(node)
                        
            # Update height information
            node.height = max(node.left_height, node.right_height) + 1
                
    def right_rotate(self, node):
        
        # The right child of the node becomes the parent
        if node.right is None:
            raise Exception("Cannot right rotate leaf node!")
            
        right = node.right
        node.right = right.left
        right.left = node
        
        # Check for the special case if we need to update the root of the tree
        if node is self.root:
            self.root = right
        
        # Otherwise update correct parents
        else:
            if node is node.parent.left:
                node.parent.left = right
            else:
                node.parent.right = right
                
            right.parent = node.parent
            node.parent = right
        
    def left_rotate(self, node):
        
        # The left child of the node becomes the parent
        if node.left is None:
            raise Exception("Cannot left rotate leaf node!")
            
        left = node.left
        node.left = left.right
        left.right = node
        
        # Check for the special case if we need to update the root of the tree
        if node is self.root:
            self.root = left
            
        # Otherwise update correct parents
        else:
            if node is node.parent.left:
                node.parent.left = left
            else:
                node.parent.right = left
                
            left.parent = node.parent
            node.parent = left
            
    def find_tightest_lower_bound(self, z, transformer=lambda x: x):

        if self.root is None:
            raise Exception("Nothing in the tree!")

        else:
            # Initialise the lower bound
            lower_bound = -np.inf
            lower_bound_index = None

            current = self.root

            while current is not None:

                t = transformer(current.val)

                # Stepping right
                if t <= z:
                    lower_bound = t
                    lower_bound_index = current.idx

                    current = current.right

                # Stepping left
                else:
                    current = current.left

            return lower_bound, lower_bound_index

    def _pretty_print(self, node, indent):

        if node is not None:

            self._pretty_print(node.left, indent + 2)

            print("  " * indent + " " + str(node),)

            self._pretty_print(node.right, indent + 2)

    def pretty_print(self):
        self._pretty_print(self.root, 0)