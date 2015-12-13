import pytest
from learningALE.handlers.binarytree import Node, BinaryTree


@pytest.fixture(scope='module')
def node():
    root = Node(0)
    return root


def test_insert(node: Node):
    assert node.right is None and node.left is None, "no inserts can be made before this test"

    # right insert
    node.insert(1)
    assert isinstance(node.right, Node)
    assert node.right.value == 1

    # left insert
    node.insert(-1)
    assert isinstance(node.left, Node)
    assert node.left.value == -1


def test_depth(node: Node):
    assert node.depth() == 2


def test_size(node: Node):
    assert node.get_size() == 3


def test_yx_vals(node: Node):
    yx_list = node.get_yx_vals([], 2, 2)
    assert yx_list[0] == [0, 1, 0]
    assert yx_list[1] == [1, 2, 1]
    assert yx_list[2] == [1, 0, -1]


def test_pop_max(node: Node):
    node_left = node.left

    node_val, _, _, _ = node.pop_max()
    assert node_val == 1

    node_val, _, hanging_left, terminal = node.pop_max()
    assert terminal == 1
    assert hanging_left == node_left
    assert node_val == 0


def test_binary_tree_pop_max():
    btree = BinaryTree()
    btree.insert(0)
    btree.insert(1)
    btree.insert(-1)
    btree.pop_max()
    btree.pop_max()
    # binary tree handles hanging left implicitly so we need to test it works
    assert btree.root.value == -1


