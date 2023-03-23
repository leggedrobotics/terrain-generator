import trimesh
import random
import numpy as np
from utils import get_heights_from_mesh
from typing import Tuple
from alive_progress import alive_bar


class LSystem:
    def __init__(self, axiom, rules, angle_adjustment=22.5, num_branches=2):
        self.axiom = axiom
        self.rules = rules
        self.angle_adjustment = angle_adjustment
        self.num_branches = num_branches

    def generate(self, iterations):
        state = self.axiom
        for i in range(iterations):
            new_state = ""
            for c in state:
                if c in self.rules:
                    rule = self.rules[c]
                    if "{" in rule:
                        # Choose random rotation directions and substitute them into the rule
                        directions = random.sample(["+", "-", "^", "&", "\\", "/"], self.num_branches + 2)
                        rule_args = [directions[0], directions[1]] + [
                            f"X{{{i}}}" for i in range(2, self.num_branches + 2)
                        ]
                        rule = rule.format(*rule_args)
                    else:
                        rule = rule.format(self.angle_adjustment)
                    new_state += rule
                else:
                    new_state += c
            state = new_state
        return state


def generate_tree_mesh(
    num_branches: int = 2, iterations: int = 3, angle_adjustment: float = 22.5, cylinder_sections: int = 8
):
    """
    Generate a tree mesh using an L-system.
    Args:
        num_branches: Number of branches to generate
        iterations: Number of iterations to run the L-system
        angle_adjustment: Angle adjustment for each iteration
        cylinder_sections: Number of sections to use for the tree cylinders
    Returns:
        tree_mesh: A trimesh.Trimesh object representing the tree mesh
    """
    rules = {
        "F": "FF",
        "X": "F[{0}X]{1}"
        + "".join([f"[{0}+X{{{i}}}]" for i in range(2, 2 + num_branches - 2)])
        + "".join([f"[{0}-X{{{i}}}]" for i in range(2 + num_branches - 2, 2 + 2 * (num_branches - 2))]),
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }
    rules = {
        "F": "FF",
        "X": "F[{0}X]{1}"
        + "".join([f"[{0}+X{{{i}}}]" for i in range(2, 2 + num_branches - 2)])
        + "".join([f"[{0}-X{{{i}}}]" for i in range(2 + num_branches - 2, 2 + 2 * (num_branches - 2))]),
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }
    rules = {
        "F": "FF",
        "X": "F[+X]F[-X]{0}" + "".join([f"[{0}+X{{{i}}}][{0}-X{{{i}}}]" for i in range(2, 2 + num_branches - 2)]),
        "+": "+{0}",
        "-": "-{0}",
    }

    # Generate the L-system state
    lsys = LSystem("X", rules, angle_adjustment=angle_adjustment, num_branches=num_branches)
    state = lsys.generate(iterations)

    # Initialize the tree mesh
    tree = trimesh.Trimesh()

    # Set the initial position and orientation of the turtle
    position = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])
    rot_mat = np.eye(4)
    stack = []

    # Define the turtle movement parameters
    step_size = 0.1
    angle = 20.7 * (np.pi / 180.0)

    # Generate the tree mesh
    cylinder_list = []
    for c in state:
        if c == "F":
            # Move the turtle forward and add a cylinder to the list
            endpoint = position + step_size * direction
            distance = np.linalg.norm(endpoint)
            radius = max(0.005, min(0.04 - 0.02 * distance, 0.03))
            cylinder = trimesh.creation.cylinder(radius=radius, height=step_size, sections=cylinder_sections)
            center = (position + endpoint) / 2.0
            cylinder.apply_transform(rot_mat)
            cylinder.apply_translation(center)
            cylinder_list.append(cylinder)
            position = endpoint
        elif c == "+":
            # Rotate the turtle around the X axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        elif c == "-":
            # Rotate the turtle around the X axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [1, 0, 0])
        elif c == "&":
            # Rotate the turtle around the Y axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        elif c == "^":
            # Rotate the turtle around the Y axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [0, 1, 0])
        elif c == "\\":
            # Rotate the turtle around the Z axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        elif c == "/":
            # Rotate the turtle around the Z axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [0, 0, 1])
        elif c == "[":
            # Push the turtle position and direction onto the stack
            stack.append((position, direction, rot_mat))
        elif c == "]":
            # Pop the turtle position and direction from the stack
            position, direction, rot_mat = stack.pop()
        else:
            pass
        if c in ["+", "-", "&", "^", "\\", "/"]:
            # Rotate the turtle around the X axis
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)

    tree = trimesh.util.concatenate(cylinder_list)
    # Smooth the tree mesh
    tree = tree.smoothed()

    # Scale the tree mesh to a realistic size
    scale_factor = 10.0
    tree.apply_scale(scale_factor)

    return tree


def add_trees_on_terrain(
    terrain_mesh: trimesh.Trimesh,
    num_trees: int = 10,
    tree_scale_range: Tuple[float, float] = (0.5, 1.5),
    tree_deg_range: Tuple[float, float] = (-30.0, 30.0),
    tree_cylinder_sections: int = 6,
):
    """
    Add trees to a terrain mesh.
    Args:
        terrain_mesh: A trimesh.Trimesh object representing the terrain mesh
        num_trees: Number of trees to add
        tree_scale_range: Range of tree scales to use
        tree_deg_range: Range of tree rotation angles to use
        tree_cylinder_sections: Number of sections to use for the tree cylinders
    Returns:
        A trimesh.Trimesh object representing the terrain mesh with trees
    """

    # Generate a tree mesh using the provided function

    bbox = terrain_mesh.bounding_box.bounds
    tree_meshes = []
    # apply random rotations and translations to the tree meshes
    positions = np.zeros((num_trees, 3))
    positions[:, 0] = np.random.uniform(bbox[0][0], bbox[1][0], size=(num_trees,))
    positions[:, 1] = np.random.uniform(bbox[0][1], bbox[1][1], size=(num_trees,))
    positions[:, 2] = get_heights_from_mesh(terrain_mesh, positions[:, :2])

    tree_rad_range = (tree_deg_range[0] * np.pi / 180.0, tree_deg_range[1] * np.pi / 180.0)

    with alive_bar(num_trees, dual_line=True, title="tree generation") as bar:
        for i in range(num_trees):
            num_branches = np.random.randint(2, 4)
            tree_mesh = generate_tree_mesh(num_branches=num_branches, cylinder_sections=tree_cylinder_sections)
            tree_mesh.apply_scale(np.random.uniform(*tree_scale_range))
            pose = np.eye(4)
            pose[:3, 3] = positions[i]
            q = trimesh.transformations.quaternion_from_euler(
                np.random.uniform(tree_rad_range[0], tree_rad_range[1]),
                np.random.uniform(tree_rad_range[0], tree_rad_range[1]),
                np.random.uniform(0, 2 * np.pi),
            )
            pose[:3, :3] = trimesh.transformations.quaternion_matrix(q)[:3, :3]
            tree_mesh.apply_transform(pose)
            tree_meshes.append(tree_mesh)
            bar()

    # Merge all the tree meshes into a single mesh
    tree_mesh = trimesh.util.concatenate(tree_meshes)

    return tree_meshes


if __name__ == "__main__":
    # Generate the tree mesh and export it to an OBJ file
    tree_mesh = generate_tree_mesh(num_branches=4, cylinder_sections=6)
    tree_mesh.show()
    tree_mesh.export("tree.obj")
