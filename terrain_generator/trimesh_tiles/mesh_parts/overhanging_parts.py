import trimesh
import numpy as np
from ...utils import (
    merge_meshes,
    yaw_rotate_mesh,
    rotate_mesh,
    ENGINE,
    get_height_array_of_mesh,
    get_heights_from_mesh,
)
from .mesh_parts_cfg import (
    FloatingBoxesPartsCfg,
    WallMeshPartsCfg,
    OverhangingMeshPartsCfg,
    FloatingBoxesPartsCfg,
    PlatformMeshPartsCfg,
    # StairMeshPartsCfg,
)


def create_wall(width, height, depth):
    mesh = trimesh.creation.box([width, height, depth])
    return mesh


# TODO: finish this.
def create_horizontal_bar(width, height, depth):
    # mesh = trimesh.creation.box([width, height, depth])
    mesh = trimesh.creation.cylinder()
    return mesh


def generate_wall_from_array(cfg: WallMeshPartsCfg) -> trimesh.Trimesh:
    """generate wall mesh from connection array.
    Args: connection_array (np.ndarray) shape=(3, 3)
    Return: trimesh.Trimesh
    """
    assert cfg.connection_array.shape[0] == 3 and cfg.connection_array.shape[1] == 3
    # If all connection is 0, return empty mesh
    if cfg.connection_array.sum() == 0:
        return trimesh.Trimesh()

    grid_size = cfg.dim[0] / cfg.connection_array.shape[0]
    meshes = []
    wall_fn = create_wall
    for y in range(cfg.connection_array.shape[1]):
        for x in range(cfg.connection_array.shape[0]):
            if cfg.connection_array[x, y] > 0:
                pos = np.array([x * grid_size, y * grid_size, 0])
                pos[:2] += grid_size / 2.0 - cfg.dim[0] / 2.0
                pos[2] += cfg.wall_height / 2.0 - cfg.dim[2] / 2.0
                if np.abs(pos[0]) > 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                    mesh = wall_fn(grid_size, cfg.wall_thickness, cfg.wall_height)
                    mesh.apply_translation(pos)
                    meshes.append(mesh)
                elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) > 1.0e-4:
                    mesh = wall_fn(cfg.wall_thickness, grid_size, cfg.wall_height)
                    mesh.apply_translation(pos)
                    meshes.append(mesh)
                elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                    # Get the coordinates of the neighboring walls
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if (
                            0 <= x + dx < cfg.connection_array.shape[0]
                            and 0 <= y + dy < cfg.connection_array.shape[1]
                            and cfg.connection_array[x + dx, y + dy] > 0
                        ):
                            neighbors.append((dx, dy))
                    if not neighbors:
                        continue

                    # Calculate the dimensions of the center wall
                    # width = grid_size + cfg.wall_thickness
                    # height = grid_size + cfg.wall_thickness
                    for dx, dy in neighbors:
                        width = grid_size / 2.0
                        height = grid_size / 2.0
                        depth = cfg.wall_height
                        p = pos.copy()
                        if dx == 1:
                            width += cfg.wall_thickness / 2.0
                            height = cfg.wall_thickness
                            p[0] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dx == -1:
                            width += cfg.wall_thickness / 2.0
                            height = cfg.wall_thickness
                            p[0] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dy == 1:
                            height += cfg.wall_thickness / 2.0
                            width = cfg.wall_thickness
                            p[1] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        elif dy == -1:
                            height += cfg.wall_thickness / 2.0
                            width = cfg.wall_thickness
                            p[1] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                        else:
                            continue

                        mesh = wall_fn(width, height, depth)
                        mesh.apply_translation(p)
                        meshes.append(mesh)
                else:
                    mesh = wall_fn(grid_size, grid_size, cfg.wall_height)
                    meshes.append(mesh)
    mesh = merge_meshes(meshes, minimal_triangles=cfg.minimal_triangles, engine=ENGINE)
    mesh = yaw_rotate_mesh(mesh, 270)  # This was required to match the connection array
    return mesh


def create_overhanging_boxes(cfg: FloatingBoxesPartsCfg, **kwargs):
    if cfg.mesh is not None:
        height_array = get_height_array_of_mesh(cfg.mesh, cfg.dim, cfg.box_grid_n)
    elif cfg.height_array is not None:
        height_array = cfg.height_array
    else:
        height_array = np.zeros((cfg.box_grid_n, cfg.box_grid_n))
    array = np.random.normal(cfg.gap_mean, cfg.gap_std, size=height_array.shape)
    array += height_array
    z_dim_array = np.ones_like(array) * cfg.box_height
    floating_array = array + z_dim_array

    overhanging_cfg = PlatformMeshPartsCfg(
        name="floating_boxes",
        dim=cfg.dim,
        array=floating_array,
        z_dim_array=z_dim_array,
        rotations=(90, 180, 270),
        flips=("x", "y"),
        weight=0.1,
        minimal_triangles=False,
        add_floor=False,
        use_z_dim_array=True,
    )
    return overhanging_cfg


def create_table_mesh(top_size=(1.0, 1.0, 0.05), leg_size=(0.05, 0.05, 0.5), leg_positions=None):
    if leg_positions is None:
        leg_positions = [
            (-top_size[0] / 2.0 + leg_size[0] / 2.0, -top_size[1] / 2.0 + leg_size[1] / 2.0),
            (-top_size[0] / 2.0 + leg_size[0] / 2.0, top_size[1] / 2.0 - leg_size[1] / 2.0),
            (top_size[0] / 2.0 - leg_size[0] / 2.0, -top_size[1] / 2.0 + leg_size[1] / 2.0),
            (top_size[0] / 2.0 - leg_size[0] / 2.0, top_size[1] / 2.0 - leg_size[1] / 2.0),
        ]
        # leg_positions = [(-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)]

    table_top = trimesh.creation.box(extents=top_size)
    # table_top.visual.face_colors = [100, 100, 255, 150]

    table_legs = []
    for pos in leg_positions:
        leg = trimesh.creation.box(extents=leg_size)
        # leg.visual.face_colors = [100, 100, 255, 50]
        leg.apply_translation((pos[0], pos[1], -leg_size[2] / 2))
        table_legs.append(leg)

    table = trimesh.util.concatenate([table_top, *table_legs])
    table = table.apply_translation((0, 0, leg_size[2] + top_size[2] / 2))

    return table


def create_archway_mesh(
    array=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    width=2.0,
    depth=2.0,
    height=1.0,
    # thickness=0.2,
    radius=None,
):
    if radius is None:
        radius = height / 2

    mesh = trimesh.creation.box(extents=(width, depth, height))
    if array[0, 1] > 0.0:
        cylinder = trimesh.creation.cylinder(radius=radius, height=width / 2.0 + radius / 2.0)
        cylinder = rotate_mesh(cylinder, 90, [1, 0, 0])
        cylinder.apply_translation((0, -width / 4.0, -height / 2))
        mesh = trimesh.boolean.difference([mesh, cylinder])
    if array[-1, 1] > 0.0:
        cylinder = trimesh.creation.cylinder(radius=radius, height=width / 2.0 + 0.1 + radius / 2.0)
        cylinder = rotate_mesh(cylinder, 90, [1, 0, 0])
        cylinder.apply_translation((0.0, width / 4.0, -height / 2))
        mesh = trimesh.boolean.difference([mesh, cylinder])
    if array[1, 0] > 0.0:
        cylinder = trimesh.creation.cylinder(radius=radius, height=width / 2.0 + 0.1 + radius / 2.0)
        cylinder = rotate_mesh(cylinder, 90, [1, 0, 0])
        cylinder = rotate_mesh(cylinder, 90, [0, 0, 1])
        cylinder.apply_translation((-width / 4.0, 0, -height / 2))
        mesh = trimesh.boolean.difference([mesh, cylinder])
    if array[1, -1] > 0.0:
        cylinder = trimesh.creation.cylinder(radius=radius, height=width / 2.0 + 0.1 + radius / 2.0)
        cylinder = rotate_mesh(cylinder, 90, [1, 0, 0])
        cylinder = rotate_mesh(cylinder, 90, [0, 0, 1])
        cylinder.apply_translation((width / 4.0, 0, -height / 2))
        mesh = trimesh.boolean.difference([mesh, cylinder])

    # archway = trimesh.boolean.difference([box, cylinder])
    # archway.apply_translation((0, -thickness / 2, 0))
    # archway.apply_translation((0, -thickness / 2, 0))
    return mesh


# def create_archway(radius=0.5, height=1.0, num_segments=10):
#     arch = trimesh.creation.cylinder(radius=radius, height=height, sections=num_segments)
#     arch.apply_scale([1, height / (2 * radius), height / (2 * radius)])
#     arch.apply_translation([0, 0, height / 2])
#     # arch.apply_rotation([0, 0, np.pi / 2], point=[0, 0, 0])
#     return arch


def create_irregular_overhang_mesh(vertices, height=0.5):
    irregular_overhang = trimesh.creation.convex_hull(vertices)
    irregular_overhang.apply_translation((0, 0, height))
    return irregular_overhang


if __name__ == "__main__":
    # table = create_table_mesh()
    # table.show()

    # vertices = [(-0.5, 0, 0), (0.5, 0, 0), (0.2, 0.2, 0.2), (-0.2, 0.2, 0.2), (0, 0.4, 0.3)]
    #
    # irregular_overhang = create_irregular_overhang_mesh(vertices, height=0.5)
    # irregular_overhang.show()

    archway = create_archway_mesh(array=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    archway.show()

    archway = create_archway_mesh(array=np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    archway.show()
