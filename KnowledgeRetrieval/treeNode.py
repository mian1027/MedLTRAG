class Node:
    def __init__(
        self,
        header: str = "",
        relation: str = "",
        trail: str = "",
        headRelation: bool = False,
        firstMatch: str = "",
        lastMatch: str = "",
        match_kg: list = None,
    ):
        self.firstMatch = firstMatch
        self.lastMatch = lastMatch
        self.headRelation = headRelation
        self.header = header
        self.match_kg = match_kg if match_kg is not None else []
        self.relation = relation
        self.trail = trail

        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def paths(self):
        return f"{self.header}->{self.relation}->{self.trail}"


class MultiPathLinkedList:
    def __init__(self, root: Node):
        self.root = root

    def _get_label(self, node):
        return f"{node.header}->{node.relation}->{node.trail}"

    def print_paths_from_root_to_leaves(self):
        def dfs(node, path):
            path.append(self._get_label(node))
            if not node.children:
                print(" -> ".join(path))
            else:
                for child in node.children:
                    dfs(child, path[:])
        dfs(self.root, [])

    def print_all_local_subpaths(self):
        all_paths = []
        def dfs_local(node):
            if not node.children:
                node_path = self._get_label(node)
                if node_path not in all_paths:
                    all_paths.append(node_path)
                return
            for child in node.children:
                node_path = self._get_label(node)
                if node_path not in all_paths:
                    all_paths.append(node_path)
                dfs_local(child)

        dfs_local(self.root)
        return all_paths

