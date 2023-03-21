# import trimesh
# import numpy as np
# import random
#
#
# class LSystem:
#     def __init__(self, axiom, rules):
#         self.axiom = axiom
#         self.rules = rules
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     new_state += self.rules[c]
#                 else:
#                     new_state += c
#             state = new_state
#         return state
#
#
# def generate_tree_mesh():
#     # Define the L-system rules
#     rules = {"F": "FF", "X": "F-[[X]+X]+F[+FX]-X"}
#
#     # Generate the L-system state
#     lsys = LSystem("X", rules)
#     state = lsys.generate(2)
#     print("state ", state, len(state))
#
#     # Initialize the tree mesh
#     tree = trimesh.Trimesh()
#
#     # Set the initial position and orientation of the turtle
#     position = np.array([0, 0, 0])
#     direction = np.array([0, 0, 1])
#     stack = []
#
#     # Define the turtle movement parameters
#     step_size = 0.1
#     angle = 25.7 * (3.14159265358979323846 / 180.0)
#
#     # Generate the tree mesh
#     for c in state:
#         print("c", c)
#         if c == "F":
#             # Move the turtle forward and add a cylinder to the mesh
#             endpoint = [position[i] + step_size * direction[i] for i in range(3)]
#             cylinder = trimesh.creation.cylinder(radius=0.01, height=step_size)
#             trans_mat = np.eye(4)
#             trans_mat[:3, 3] = position
#             z_axis = direction / np.linalg.norm(direction)
#             x_axis = np.cross([0, 0, 1], z_axis)
#             x_axis /= np.linalg.norm(x_axis)
#             y_axis = np.cross(z_axis, x_axis)
#             y_axis /= np.linalg.norm(y_axis)
#             # rot_mat = np.eye(4)
#             # rot_mat[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=1)
#             # trans_mat = np.dot(trans_mat, rot_mat)
#
#             # Apply the transformation matrix to the cylinder
#             cylinder.apply_transform(trans_mat)
#             # cylinder.apply_translation(position)
#             tree = tree + cylinder
#             position = endpoint
#         elif c == "+":
#             # Rotate the turtle around the X axis
#             rot_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
#             direction = np.dot(rot_matrix[:3, :3], direction)
#         elif c == "-":
#             # Rotate the turtle around the X axis
#             rot_matrix = trimesh.transformations.rotation_matrix(-angle, [1, 0, 0])
#             direction = np.dot(rot_matrix[:3, :3], direction)
#         elif c == "[":
#             # Push the turtle position and direction onto the stack
#             stack.append((position, direction))
#         elif c == "]":
#             # Pop the turtle position and direction from the stack
#             position, direction = stack.pop()
#         else:
#             pass
#
#     # Smooth the tree mesh
#     tree = tree.smoothed()
#
#     # Scale the tree mesh to a realistic size
#     scale_factor = 10.0
#     tree.apply_scale(scale_factor)
#
#     return tree

import trimesh
import random
import numpy as np


# class LSystem:
#     def __init__(self, axiom, rules):
#         self.axiom = axiom
#         self.rules = rules
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     new_state += self.rules[c]
#                 else:
#                     new_state += c
#             state = new_state
#         return state


# class LSystem:
#     def __init__(self, axiom, rules):
#         self.axiom = axiom
#         self.rules = rules
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     rule = self.rules[c]
#                     if "{" in rule:
#                         # Choose two random rotation directions and substitute them into the rule
#                         directions = random.sample(["+", "-", "^", "&", "\\", "/"], 2)
#                         rule = rule.format(directions[0], directions[1])
#                     new_state += rule
#                 else:
#                     new_state += c
#             state = new_state
#         return state


# class LSystem:
#     def __init__(self, axiom, rules):
#         self.axiom = axiom
#         self.rules = rules
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     rule = self.rules[c]
#                     if "{" in rule:
#                         # Choose two random rotation directions and substitute them into the rule
#                         directions = random.sample(["+", "-", "^", "&", "\\", "/"], 2)
#                         rule = random.choice(
#                             [rule.format(directions[0], directions[1]), rule.format(directions[1], directions[0])]
#                         )
#                     new_state += rule
#                 else:
#                     new_state += c
#             state = new_state
#         return state


# class LSystem:
#     def __init__(self, axiom, rules, angle_adjustment=0.0):
#         self.axiom = axiom
#         self.rules = rules
#         self.angle_adjustment = angle_adjustment
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     rule = self.rules[c]
#                     if "{" in rule:
#                         # Choose two random rotation directions and substitute them into the rule
#                         directions = random.sample(["+", "-", "^", "&", "\\", "/"], 2)
#                         # Add the angle adjustment factors to the rule
#                         rule = random.choice(
#                             [
#                                 rule.format(
#                                     directions[0], directions[1], self.angle_adjustment, -self.angle_adjustment
#                                 ),
#                                 rule.format(
#                                     directions[1], directions[0], -self.angle_adjustment, self.angle_adjustment
#                                 ),
#                             ]
#                         )
#                     new_state += rule
#                 else:
#                     new_state += c
#             state = new_state
#         return state


# class LSystem:
#     def __init__(self, axiom, rules, angle_adjustment=0.0, num_branches=3):
#         self.axiom = axiom
#         self.rules = rules
#         self.angle_adjustment = angle_adjustment
#         self.num_branches = num_branches
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     rule = self.rules[c]
#                     if "{" in rule:
#                         # Choose two random rotation directions and substitute them into the rule
#                         directions = random.sample(["+", "-", "^", "&", "\\", "/"], self.num_branches)
#                         # Add the angle adjustment factors to the rule
#                         rule_args = [directions[0], directions[1]] + [self.angle_adjustment] * (self.num_branches - 2)
#                         rule = rule.format(*rule_args)
#                     new_state += rule
#                 else:
#                     new_state += c
#             state = new_state
#         return state


# class LSystem:
#     def __init__(self, axiom, rules, angle_adjustment=10, num_branches=2):
#         self.axiom = axiom
#         self.rules = rules
#         self.num_branches = num_branches
#
#     def generate(self, iterations):
#         state = self.axiom
#         for i in range(iterations):
#             new_state = ""
#             for c in state:
#                 if c in self.rules:
#                     rule = self.rules[c]
#                     if "{" in rule:
#                         # Choose random rotation directions and substitute them into the rule
#                         directions = random.sample(["+", "-", "^", "&", "\\", "/"], self.num_branches + 2)
#                         rule_args = [directions[0], directions[1]] + [
#                             f"X{{{i}}}" for i in range(2, self.num_branches + 2)
#                         ]
#                         rule = rule.format(*rule_args)
#                     new_state += rule
#                 else:
#                     new_state += c
#             state = new_state
#         return state


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


def generate_tree_mesh():
    # Define the L-system rules
    # rules = {"F": "FF", "X": "F-[[X]+X]+F[+FX]-X"}
    # Define the L-system rules
    # rules = {
    #     "F": "FF",
    #     "X": "F-[[X]+X]+F[+FX]-X",
    #     "+": "v[/&[+\\^-]^-]-[^/\\&[-v+]+v]",
    #     "-": "v[+&[-^/]+^]-[^\\&[+v-]v+]",
    # }
    rules = {
        "F": "FF",
        "X": "F-[[X]+X]+F[+FX]-X",
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }
    rules = {
        "F": "FF",
        "X": "F[{0}X]{1}[{0}+X]{1}[{0}-X]{1}",
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }

    rules = {
        "F": "FF",
        # "X": "F[{0}X]{1}[{0}+X]{1}[{0}-X]{1}",
        "X": "F[{0}X]{1}[{0}+X{2}]{1}[{0}-X{3}]{1}",
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }
    rules = {
        "F": "FF",
        "X": "F[{0}X]{1}[{0}+X{2}]{1}[{0}-X{3}]{1}",
        "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
        "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    }
    # rules = {
    #     "F": "FF",
    #     "X": "F[{0}X]{1}"
    #     + "".join([f"[{0}+X{{{i}}}]" for i in range(2, 2 + 4)])
    #     + "".join([f"[{0}-X{{{i}}}]" for i in range(2 + 4, 2 + 4 + 4)]),
    #     "+": "v[/&[+\\^-]{0}]{1}-[{0}]^/\\&[-v+]+v",
    #     "-": "v[+&[-^{1}/]+^{0}]-{0}\\&[+v-]v+",
    # }
    num_branches = 4
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
    lsys = LSystem("X", rules, angle_adjustment=20.5, num_branches=num_branches)
    state = lsys.generate(3)

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
            # endpoint = [position[i] + step_size * direction[i] for i in range(3)]
            endpoint = position + step_size * direction
            print("end point ", endpoint)
            distance = np.linalg.norm(endpoint)
            radius = max(0.005, min(0.04 - 0.02 * distance, 0.03))
            # radius = max(0.01, 0.03 - 0.02 * endpoint[2])
            # radius_list.append(radius)
            # height_list.append(step_size)
            cylinder = trimesh.creation.cylinder(radius=radius, height=step_size)
            center = (position + endpoint) / 2.0
            cylinder.apply_transform(rot_mat)
            cylinder.apply_translation(center)
            cylinder_list.append(cylinder)
            position = endpoint
            # tree = tree + cylinder
        elif c == "+":
            # Rotate the turtle around the X axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
            # rot_mat = np.dot(rot_mat, rot_matrix)
        elif c == "-":
            # Rotate the turtle around the X axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [1, 0, 0])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif c == "&":
            # Rotate the turtle around the Y axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif c == "^":
            # Rotate the turtle around the Y axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [0, 1, 0])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif c == "\\":
            # Rotate the turtle around the Z axis
            rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif c == "/":
            # Rotate the turtle around the Z axis
            rot_matrix = trimesh.transformations.rotation_matrix(-angle, [0, 0, 1])
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif c == "[":
            # Push the turtle position and direction onto the stack
            stack.append((position, direction, rot_mat))
        elif c == "]":
            # Pop the turtle position and direction from the stack
            position, direction, rot_mat = stack.pop()
        else:
            pass

    tree = trimesh.util.concatenate(cylinder_list)
    # Smooth the tree mesh
    tree = tree.smoothed()

    # Scale the tree mesh to a realistic size
    scale_factor = 10.0
    tree.apply_scale(scale_factor)

    return tree


# Generate the tree mesh and export it to an OBJ file
tree_mesh = generate_tree_mesh()
tree_mesh.show()
tree_mesh.export("tree.obj")
