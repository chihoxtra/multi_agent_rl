import numpy as np


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py

    Here we have 2 data object:
    - Tree: a linear numpy array to store all td values. td values are stored
    at lowest leafs. tree[0:capacity-1] are parents nodes tds. tree[capacity:end]
    stores lowest leafs data.
    - data: store the experience data object. the latest next slot to store are
    stored in data pointer. FIFO. Conversion between data pointer and tree index
    are:
        leaf_index = self.data_pointer + self.capacity - 1
    """
    data_pointer = 0 #this is to track which slot should we update next

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of memories allowed in the tree

        # Generate the tree with all nodes values (td error) = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Each node as max 2 children so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes (sum of children td values) range from 0 : capacity - 1
        # Leaf nodes (all td errors values) range from capacity : end
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the td_score
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)


    """
    Here we add our td_score or the td error in the sumtree leaf (self.tree)
    and add the experience in to self.data.
    """
    def add(self, td_score, data):
        # which leaf node we want to store the data
        leaf_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(leaf_index, td_score)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If > the capacity, restart and overwrite
            self.data_pointer = 0


    """
    Update the leaf td score and propagate the change through tree
    """
    def update(self, leaf_index, td_score):
        # Change = new td_score score - former td_score
        change = td_score - self.tree[leaf_index]
        self.tree[leaf_index] = td_score

        # then propagate the change through tree
        while leaf_index != 0:    # propagate the changes until it reaches root

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE TD SCORE VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6]

            If we are in leaf at index 6, we updated the td_score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            leaf_index = (leaf_index - 1) // 2
            self.tree[leaf_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index #get the index of closest value
                break

            else: # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_td_score(self):
        return self.tree[0] # Returns the root node
