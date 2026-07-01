import re
import time
from math import *

import gol
import gurobipy
import numpy as np
from jerk import *
from shapely.geometry import LinearRing, LineString, Point, Polygon

gurobipy.setParam("LogToConsole", 0)
gurobipy.setParam("OutputFlag", 0)


def CtoAB(A, B, C):
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C_np = np.array(C, dtype=np.float64)
    CA = A_np - C_np
    CB = B_np - C_np
    ABdistance = np.linalg.norm(A_np - B_np)
    return abs(np.cross(CA, CB)) / ABdistance


def generate_vector(A, B):
    return [B[0] - A[0], B[1] - A[1]]


def generate_normal_vector(A, B):
    length = sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
    return [(B[0] - A[0]) / length, (B[1] - A[1]) / length]


def inclination_angle(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    return atan2(dy, dx)


def calculate_intersect_angle_AB_CD(A, B, C, D):
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C_np = np.array(C, dtype=np.float64)
    D_np = np.array(D, dtype=np.float64)

    AB = B_np - A_np
    CD = D_np - C_np

    ab_norm = np.linalg.norm(AB)
    cd_norm = np.linalg.norm(CD)

    dot_prod = np.dot(AB, CD)
    cross_prod = np.cross(AB, CD)

    norm_product = ab_norm * cd_norm
    if norm_product == 0:
        return 0.0

    cos_angle = dot_prod / norm_product
    sin_angle = cross_prod / norm_product

    if abs(dot_prod) < norm_product:
        tar_angle = acos(cos_angle) if sin_angle >= 0 else -acos(cos_angle)
        return tar_angle * 180 / pi
    else:
        return 0.0


def calculate_intersect_point_AB_CD(A, B, C, D):
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = (B[0] - A[0]) * A[1] - (B[1] - A[1]) * A[0]

    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = (D[0] - C[0]) * C[1] - (D[1] - C[1]) * C[0]

    T1 = np.array([[a1, b1], [a2, b2]], dtype=np.float64)
    T2 = np.array([-c1, -c2], dtype=np.float64)

    try:
        return np.linalg.solve(T1, T2).tolist()
    except np.linalg.LinAlgError:
        return False


def calculate_current_CoM(
    ini_vertex_position, current_vertex_position, ini_CoM_position
):
    # 1. 快速转换为numpy数组，避免手动构造矩阵
    ini_verts = np.asarray(ini_vertex_position, dtype=np.float64)  # (n, 2)
    curr_verts = np.asarray(current_vertex_position, dtype=np.float64)  # (n, 2)
    ini_CoM = np.asarray(ini_CoM_position, dtype=np.float64)  # (2,)

    # 2. 构建仿射变换的设计矩阵（添加齐次坐标列，向量化操作）
    n = ini_verts.shape[0]
    X = np.hstack([ini_verts, np.ones((n, 1), dtype=np.float64)])  # (n, 3)，无需转置

    # 3. 最小二乘求解仿射变换矩阵（替代原逆矩阵，适配任意顶点数n≥3）
    # 直接求解AX=B，A为3×2变换矩阵，避免嵌套dot和转置
    transform = np.linalg.lstsq(X, curr_verts, rcond=None)[
        0
    ].T  # (2, 3)，一步得到变换矩阵

    # 4. 计算目标重心（齐次坐标乘法，无多余类型转换）
    curr_CoM_homo = np.hstack([ini_CoM, 1.0])  # (3,)
    current_CoM = transform @ curr_CoM_homo  # (2,)，向量乘法效率最高

    return current_CoM.tolist()


def footCtoAB(A, B, C):
    Ax, Ay = A[0], A[1]
    Bx, By = B[0], B[1]
    Cx, Cy = C[0], C[1]
    numerator = (Cx - Ax) * (Bx - Ax) + (Cy - Ay) * (By - Ay)
    denominator = (Bx - Ax) ** 2 + (By - Ay) ** 2
    k = numerator / denominator if denominator != 0 else 0.0
    return [Ax + k * (Bx - Ax), Ay + k * (By - Ay)]


def calculate_parameter_translate_and_rotate_moveA(ini_A, ini_B, tar_A, tar_B):
    delta_translate = [tar_A[0] - ini_A[0], tar_A[1] - ini_A[1]]
    A_r = inclination_angle(tar_A, tar_B) - inclination_angle(ini_A, ini_B)
    if A_r > 0:
        A_r = A_r % (2 * pi)
        D_r = "counterclockwise"
    else:
        A_r = A_r % (-2 * pi)
        D_r = "clockwise"
    return delta_translate, D_r, A_r


def calculate_parameter_translate_and_rotate_moveB(ini_A, ini_B, tar_A, tar_B):
    delta_translate = [tar_B[0] - ini_B[0], tar_B[1] - ini_B[1]]
    A_r = inclination_angle(tar_A, tar_B) - inclination_angle(ini_A, ini_B)
    if A_r > 0:
        A_r = A_r % (2 * pi)
        D_r = "counterclockwise"
    else:
        A_r = A_r % (-2 * pi)
        D_r = "clockwise"
    return delta_translate, D_r, A_r


def point_position_after_rotation(current_xy, rotation_pole, desired_angle):
    current_xy = np.array(current_xy, dtype=np.float64)
    rotation_pole = np.array(rotation_pole, dtype=np.float64)
    rad = desired_angle * pi / 180
    displacement = current_xy - rotation_pole
    c, s = cos(rad), sin(rad)
    rotated_x = displacement[0] * c - displacement[1] * s
    rotated_y = displacement[0] * s + displacement[1] * c
    return [rotation_pole[0] + rotated_x, rotation_pole[1] + rotated_y]


def point_position_after_rotation_batch(current_xy, rotation_pole, desired_angle):
    """
    批量旋转函数：支持current_xy为单个点(2,)或多个点(n, 2)

    参数:
        current_xy: 待旋转的点，形状为(2,)或(n, 2)
        rotation_pole: 旋转极点，形状为(2,)
        desired_angle: 旋转角度（度），形状为(m,)

    返回:
        旋转后的点列表，形状为(m, 2)（单点输入）或(n*m, 2)（多点输入）
    """
    # 转换为numpy数组并标准化维度
    current_xy = np.asarray(current_xy, dtype=np.float64)
    rotation_pole = np.asarray(rotation_pole, dtype=np.float64).reshape(
        2
    )  # 确保旋转极点是(2,)
    desired_angle = np.asarray(desired_angle, dtype=np.float64)

    # 处理单点输入（current_xy是(2,)）
    if current_xy.ndim == 1:
        # 转换为(1, 2)以保持统一处理逻辑
        current_xy = current_xy.reshape(1, 2)
        is_single_point = True
    else:
        # 验证多点输入格式
        if current_xy.ndim != 2 or current_xy.shape[1] != 2:
            raise ValueError("current_xy必须是形状为(2,)的单点或(n, 2)的多点数组")
        is_single_point = False

    # 计算弧度（批量处理所有角度）
    rad = desired_angle * pi / 180  # 形状: (m,)

    # 计算位移向量：(n, 1, 2) - (2,) → 广播为(n, m, 2)
    displacement = current_xy[:, np.newaxis, :] - rotation_pole  # 形状: (n, m, 2)

    # 批量计算旋转矩阵元素（向量化）
    c = np.cos(rad)  # 形状: (m,)
    s = np.sin(rad)  # 形状: (m,)

    # 执行旋转计算（利用广播避免循环）
    rotated_x = displacement[..., 0] * c - displacement[..., 1] * s  # 形状: (n, m)
    rotated_y = displacement[..., 0] * s + displacement[..., 1] * c  # 形状: (n, m)

    # 计算最终坐标并调整形状
    result = np.stack(
        [rotation_pole[0] + rotated_x, rotation_pole[1] + rotated_y], axis=-1
    )  # 形状: (n, m, 2)

    # 如果是单点输入，返回(m, 2)；否则返回(n*m, 2)
    if is_single_point:
        return result[0].tolist()  # 取第0个点的所有旋转结果
    else:
        return result.reshape(-1, 2).tolist()


def calculate_CoM(current_vertex_point):
    # 将顶点列表转换为NumPy数组，形状为(n, 2)，提升访问效率
    points = np.array(current_vertex_point, dtype=np.float64)
    n = points.shape[0]

    # 向量化获取前一个顶点（最后一个顶点的前一个是第一个顶点）
    prev_points = points[np.roll(range(n), 1)]  # 等效于循环中i-1和最后一个取0的逻辑

    # 提取纬度(lat)和经度(lng)
    lat = points[:, 0]
    lng = points[:, 1]
    lat1 = prev_points[:, 0]
    lng1 = prev_points[:, 1]

    # 向量化计算fg，避免循环
    fg = (lat * lng1 - lng * lat1) / 2.0

    # 累加计算面积和重心分量（向量化求和比循环累加快得多）
    area = fg.sum()
    x = (fg * (lat + lat1) / 3.0).sum() / area
    y = (fg * (lng + lng1) / 3.0).sum() / area

    return x, y


def calculate_axial_symmetric_point_matrix(axis_A, axis_B):
    # 提取坐标值，减少列表索引访问
    Ax, Ay = axis_A[0], axis_A[1]
    Bx, By = axis_B[0], axis_B[1]

    # 预计算分子分母中重复出现的项
    dx = Ax - Bx  # (A.x - B.x)
    dy = Ay - By  # (A.y - B.y)
    dx_sq = dx**2
    dy_sq = dy**2
    denom = dx_sq + dy_sq  # 分母，共用5次

    # 避免重复计算，直接推导各参数（基于轴对称变换公式简化）
    a = (dx_sq - dy_sq) / denom
    b = (2 * dx * dy) / denom
    # 简化c的计算（提取公因子2，合并同类项）
    c = 2 * (Ax * By**2 - Ax * Ay * By - Ay * Bx * By + Ay**2 * Bx) / denom
    d = b  # d与b相等，直接复用
    e = (dy_sq - dx_sq) / denom  # e是a的相反数
    # 简化f的计算（提取公因子2，合并同类项）
    f = 2 * (-Ax * Bx * By - Ax * Ay * Bx + Ax**2 * By + Ay * Bx**2) / denom

    return a, b, c, d, e, f


def calculate_axial_symmetric_point(point, axis_A, axis_B):
    # 直接计算变换矩阵参数（避免元组解包开销）
    a, b, c, d, e, f = calculate_axial_symmetric_point_matrix(axis_A, axis_B)
    # 直接计算对称点坐标（减少中间变量）
    return [a * point[0] + b * point[1] + c, d * point[0] + e * point[1] + f]


# @jit()
def contact_points_and_normal(
    current_vertex_position,
    F1_contact_point,
    F2_contact_point,
    profile_environment,
    ini_vertex_position=None,
    F1_index=None,
    F2_index=None,
    accuracy=0.01,
):
    F1 = F1_contact_point
    F2 = F2_contact_point
    contact_index = []
    contact_list = []
    contact_normal_list = []
    contact_local_tangent_list = []
    index = 1
    for i in range(len(current_vertex_position)):
        for j in range(len(profile_environment)):
            if (
                j != 0
                and j != len(profile_environment) - 1
                and Point(current_vertex_position[i]).distance(
                    Point(profile_environment[j])
                )
                < accuracy
            ):
                contact_point = current_vertex_position[i]
                contact_edge_length0 = LineString(
                    [profile_environment[j - 1], profile_environment[j]]
                ).length
                contact_edge_length1 = LineString(
                    [profile_environment[j], profile_environment[j + 1]]
                ).length
                contact_normal0 = [
                    (profile_environment[j][1] - profile_environment[j - 1][1])
                    / contact_edge_length0,
                    -(profile_environment[j][0] - profile_environment[j - 1][0])
                    / contact_edge_length0,
                ]
                contact_normal1 = [
                    (profile_environment[j + 1][1] - profile_environment[j][1])
                    / contact_edge_length1,
                    -(profile_environment[j + 1][0] - profile_environment[j][0])
                    / contact_edge_length1,
                ]
                if (
                    Point(
                        [
                            contact_point[0] + contact_normal0[0] * 0.011,
                            contact_point[1] + contact_normal0[1] * 0.011,
                        ]
                    ).distance(Polygon(current_vertex_position))
                    != 0
                ):
                    contact_normal0 = [-contact_normal0[0], -contact_normal0[1]]
                if (
                    Point(
                        [
                            contact_point[0] + contact_normal1[0] * 0.011,
                            contact_point[1] + contact_normal1[1] * 0.011,
                        ]
                    ).distance(Polygon(current_vertex_position))
                    != 0
                ):
                    contact_normal1 = [-contact_normal1[0], -contact_normal1[1]]
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        contact_normal_list[contact_list.index(point)] = [
                            contact_normal0,
                            contact_normal1,
                        ]
                        contact_local_tangent_list[contact_list.index(point)] = [
                            profile_environment[j - 1],
                            profile_environment[j],
                            profile_environment[j + 1],
                        ]
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_normal_list.append([contact_normal0, contact_normal1])
                    contact_local_tangent_list.append(
                        [
                            profile_environment[j - 1],
                            profile_environment[j],
                            profile_environment[j + 1],
                        ]
                    )
            if (
                j != len(profile_environment) - 1
                and Point(current_vertex_position[i]).distance(
                    LineString([profile_environment[j], profile_environment[j + 1]])
                )
                < accuracy
            ):
                contact_point = current_vertex_position[i]
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_edge_length = LineString(
                        [profile_environment[j], profile_environment[j + 1]]
                    ).length
                    contact_normal = [
                        (profile_environment[j + 1][1] - profile_environment[j][1])
                        / contact_edge_length,
                        -(profile_environment[j + 1][0] - profile_environment[j][0])
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                contact_point[0] + contact_normal[0] * 0.011,
                                contact_point[1] + contact_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        contact_normal = [-contact_normal[0], -contact_normal[1]]
                    contact_normal_list.append(contact_normal)
                    contact_local_tangent_list.append(
                        [profile_environment[j], profile_environment[j + 1]]
                    )
            if (
                Point(profile_environment[j]).distance(
                    LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                )
                < accuracy
            ):
                contact_point = footCtoAB(
                    current_vertex_position[i],
                    current_vertex_position[(i + 1) % (len(current_vertex_position))],
                    profile_environment[j],
                )
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    contact_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                contact_point[0] + contact_normal[0] * 0.011,
                                contact_point[1] + contact_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        contact_normal = [-contact_normal[0], -contact_normal[1]]
                    contact_normal_list.append(contact_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
    if F1 != None:
        if F1_index == None or F1_index <= 100 and "F1" not in contact_index:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and F1 != None
                    and Point(F1).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F1,
                        )
                    )
                    contact_index.append("F1")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F1_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [F1[0] + F1_normal[0] * 0.011, F1[1] + F1_normal[1] * 0.011]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        elif ini_vertex_position != None and F1_index > 100:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and Point(F1).distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F1_contact_position = current_vertex_position[i]
                    contact_list.append(F1_contact_position)
                    contact_index.append("F1")
                    F1_angle = (F1_index % 100 - 1) * 5
                    _0, _1, A_r = calculate_parameter_translate_and_rotate(
                        ini_vertex_position[0],
                        ini_vertex_position[1],
                        current_vertex_position[0],
                        current_vertex_position[1],
                    )
                    F1_angle += A_r * 180 / pi
                    F1_normal = [-sin(F1_angle * pi / 180), cos(F1_angle * pi / 180)]
                    if (
                        Point(
                            [
                                F1_contact_position[0] + F1_normal[0] * 0.011,
                                F1_contact_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            F1_contact_position,
                            [
                                F1_contact_position[0] + 1 * cos(F1_angle * pi / 180),
                                F1_contact_position[1] + 1 * sin(F1_angle * pi / 180),
                            ],
                        ]
                    )
                    break
    if F2 != None:
        if F2_index == None or F2_index <= 100 and "F2" not in contact_index:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and F2 != None
                    and Point(F2).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F2,
                        )
                    )
                    contact_index.append("F2")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F2_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [F2[0] + F2_normal[0] * 0.011, F2[1] + F2_normal[1] * 0.011]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        elif ini_vertex_position != None and F2_index > 100:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and Point(F2).distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F2_contact_position = current_vertex_position[i]
                    contact_list.append(F2_contact_position)
                    contact_index.append("F2")
                    F2_angle = (F2_index % 100 - 1) * 5
                    _0, _1, A_r = calculate_parameter_translate_and_rotate(
                        ini_vertex_position[0],
                        ini_vertex_position[1],
                        current_vertex_position[0],
                        current_vertex_position[1],
                    )
                    F2_angle += A_r * 180 / pi
                    F2_normal = [-sin(F2_angle * pi / 180), cos(F2_angle * pi / 180)]
                    if (
                        Point(
                            [
                                F2_contact_position[0] + F2_normal[0] * 0.011,
                                F2_contact_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            F2_contact_position,
                            [
                                F2_contact_position[0] + 1 * cos(F2_angle * pi / 180),
                                F2_contact_position[1] + 1 * sin(F2_angle * pi / 180),
                            ],
                        ]
                    )
                    break
    return contact_index, contact_list, contact_normal_list, contact_local_tangent_list


# time5 = time.time()
# print(contact_points_and_normal([[0.5,0], [1.2,0], [1.2,1.5], [0.5,1.5]], [0.5,0.8], [1.2,0.9], [[0,0], [3,0]]))
# print(time.time()-time5)


def contact_points_and_normal_consider_gripper_type2(
    current_vertex_position,
    F1_position,
    F2_position,
    F1_link,
    F2_link,
    profile_environment,
    accuracy=0.01,
):
    contact_index = []
    contact_list = []
    contact_normal_list = []
    contact_local_tangent_list = []
    index = 1
    for i in range(len(current_vertex_position)):
        for j in range(len(profile_environment)):
            if (
                j != 0
                and j != len(profile_environment) - 1
                and Point(current_vertex_position[i]).distance(
                    Point(profile_environment[j])
                )
                < accuracy
            ):
                contact_point = current_vertex_position[i]
                contact_edge_length0 = LineString(
                    [profile_environment[j - 1], profile_environment[j]]
                ).length
                contact_edge_length1 = LineString(
                    [profile_environment[j], profile_environment[j + 1]]
                ).length
                contact_normal0 = [
                    (profile_environment[j][1] - profile_environment[j - 1][1])
                    / contact_edge_length0,
                    -(profile_environment[j][0] - profile_environment[j - 1][0])
                    / contact_edge_length0,
                ]
                contact_normal1 = [
                    (profile_environment[j + 1][1] - profile_environment[j][1])
                    / contact_edge_length1,
                    -(profile_environment[j + 1][0] - profile_environment[j][0])
                    / contact_edge_length1,
                ]
                if (
                    Point(
                        [
                            contact_point[0] + contact_normal0[0] * 0.011,
                            contact_point[1] + contact_normal0[1] * 0.011,
                        ]
                    ).distance(Polygon(current_vertex_position))
                    != 0
                ):
                    contact_normal0 = [-contact_normal0[0], -contact_normal0[1]]
                if (
                    Point(
                        [
                            contact_point[0] + contact_normal1[0] * 0.011,
                            contact_point[1] + contact_normal1[1] * 0.011,
                        ]
                    ).distance(Polygon(current_vertex_position))
                    != 0
                ):
                    contact_normal1 = [-contact_normal1[0], -contact_normal1[1]]
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        contact_normal_list[contact_list.index(point)] = [
                            contact_normal0,
                            contact_normal1,
                        ]
                        contact_local_tangent_list[contact_list.index(point)] = [
                            profile_environment[j - 1],
                            profile_environment[j],
                            profile_environment[j + 1],
                        ]
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_normal_list.append([contact_normal0, contact_normal1])
                    contact_local_tangent_list.append(
                        [
                            profile_environment[j - 1],
                            profile_environment[j],
                            profile_environment[j + 1],
                        ]
                    )
            if (
                j != len(profile_environment) - 1
                and Point(current_vertex_position[i]).distance(
                    LineString([profile_environment[j], profile_environment[j + 1]])
                )
                < accuracy
            ):
                contact_point = current_vertex_position[i]
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_edge_length = LineString(
                        [profile_environment[j], profile_environment[j + 1]]
                    ).length
                    contact_normal = [
                        (profile_environment[j + 1][1] - profile_environment[j][1])
                        / contact_edge_length,
                        -(profile_environment[j + 1][0] - profile_environment[j][0])
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                contact_point[0] + contact_normal[0] * 0.011,
                                contact_point[1] + contact_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        contact_normal = [-contact_normal[0], -contact_normal[1]]
                    contact_normal_list.append(contact_normal)
                    contact_local_tangent_list.append(
                        [profile_environment[j], profile_environment[j + 1]]
                    )
            if (
                Point(profile_environment[j]).distance(
                    LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                )
                < accuracy
            ):
                contact_point = footCtoAB(
                    current_vertex_position[i],
                    current_vertex_position[(i + 1) % (len(current_vertex_position))],
                    profile_environment[j],
                )
                for point in contact_list:
                    if Point(contact_point).distance(Point(point)) < accuracy:
                        break
                else:
                    contact_list.append(contact_point)
                    contact_index.append("E" + str(index))
                    index = index + 1
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    contact_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                contact_point[0] + contact_normal[0] * 0.011,
                                contact_point[1] + contact_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        contact_normal = [-contact_normal[0], -contact_normal[1]]
                    contact_normal_list.append(contact_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )

    if F1_link.distance(Polygon(current_vertex_position)) < accuracy:
        if Point(F1_position).distance(Polygon(current_vertex_position)) < accuracy:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and Point(F1_position).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F1_position,
                        )
                    )
                    contact_index.append("F1")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F1_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                F1_position[0] + F1_normal[0] * 0.011,
                                F1_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        else:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and F1_link.distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F1_contact_position = current_vertex_position[i]
                    contact_list.append(F1_contact_position)
                    contact_index.append("F1")
                    F1_normal = generate_normal_vector(
                        F1_link.coords[:][0], F1_link.coords[:][1]
                    )
                    F1_normal = [-F1_normal[1], F1_normal[0]]
                    if (
                        Point(
                            [
                                F1_contact_position[0] + F1_normal[0] * 0.011,
                                F1_contact_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [list(F1_link.coords[:][0]), list(F1_link.coords[:][1])]
                    )
                    break
    if F2_link.distance(Polygon(current_vertex_position)) < accuracy:
        if Point(F2_position).distance(Polygon(current_vertex_position)) < accuracy:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and Point(F2_position).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F2_position,
                        )
                    )
                    contact_index.append("F2")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F2_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                F2_position[0] + F2_normal[0] * 0.011,
                                F2_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        else:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and F2_link.distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F2_contact_position = current_vertex_position[i]
                    contact_list.append(F2_contact_position)
                    contact_index.append("F2")
                    F2_normal = generate_normal_vector(
                        F2_link.coords[:][0], F2_link.coords[:][1]
                    )
                    F2_normal = [-F2_normal[1], F2_normal[0]]
                    if (
                        Point(
                            [
                                F2_contact_position[0] + F2_normal[0] * 0.011,
                                F2_contact_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [list(F2_link.coords[:][0]), list(F2_link.coords[:][1])]
                    )
                    break
    return contact_index, contact_list, contact_normal_list, contact_local_tangent_list


def contact_points_and_normal_finger(
    current_vertex_position,
    F1,
    F2,
    accuracy=0.01,
    ini_vertex_position=None,
    F1_index=None,
    F2_index=None,
):
    contact_index = []
    contact_list = []
    contact_normal_list = []
    contact_local_tangent_list = []
    if F1 != None:
        if F1_index == None or F1_index <= 100 and "F1" not in contact_index:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and F1 != None
                    and Point(F1).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F1,
                        )
                    )
                    contact_index.append("F1")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F1_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [F1[0] + F1_normal[0] * 0.011, F1[1] + F1_normal[1] * 0.011]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        elif ini_vertex_position != None and F1_index > 100:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and Point(F1).distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F1_contact_position = current_vertex_position[i]
                    contact_list.append(F1_contact_position)
                    contact_index.append("F1")
                    F1_angle = (F1_index % 100 - 1) * 5
                    _0, _1, A_r = calculate_parameter_translate_and_rotate(
                        ini_vertex_position[0],
                        ini_vertex_position[1],
                        current_vertex_position[0],
                        current_vertex_position[1],
                    )
                    F1_angle += A_r * 180 / pi
                    F1_normal = [-sin(F1_angle * pi / 180), cos(F1_angle * pi / 180)]
                    if (
                        Point(
                            [
                                F1_contact_position[0] + F1_normal[0] * 0.011,
                                F1_contact_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            F1_contact_position,
                            [
                                F1_contact_position[0] + 1 * cos(F1_angle * pi / 180),
                                F1_contact_position[1] + 1 * sin(F1_angle * pi / 180),
                            ],
                        ]
                    )
                    break
    if F2 != None:
        if F2_index == None or F2_index <= 100 and "F2" not in contact_index:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and F2 != None
                    and Point(F2).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F2,
                        )
                    )
                    contact_index.append("F2")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F2_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [F2[0] + F2_normal[0] * 0.011, F2[1] + F2_normal[1] * 0.011]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        elif ini_vertex_position != None and F2_index > 100:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and Point(F2).distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F2_contact_position = current_vertex_position[i]
                    contact_list.append(F2_contact_position)
                    contact_index.append("F2")
                    F2_angle = (F2_index % 100 - 1) * 5
                    _0, _1, A_r = calculate_parameter_translate_and_rotate(
                        ini_vertex_position[0],
                        ini_vertex_position[1],
                        current_vertex_position[0],
                        current_vertex_position[1],
                    )
                    F2_angle += A_r * 180 / pi
                    F2_normal = [-sin(F2_angle * pi / 180), cos(F2_angle * pi / 180)]
                    if (
                        Point(
                            [
                                F2_contact_position[0] + F2_normal[0] * 0.011,
                                F2_contact_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            F2_contact_position,
                            [
                                F2_contact_position[0] + 1 * cos(F2_angle * pi / 180),
                                F2_contact_position[1] + 1 * sin(F2_angle * pi / 180),
                            ],
                        ]
                    )
                    break
    return contact_index, contact_list, contact_normal_list, contact_local_tangent_list


# print contact_points_and_normal([[0.5, 0.0], [1.2, 0.0], [1.2, 1.5], [0.5, 1.5]], [0.5, 1.298], [1.2, 0.9500000000000002], [[0,0], [3,0]], accuracy = 0.01)


def contact_points_and_normal_consider_gripper_type2_finger(
    current_vertex_position, F1_position, F2_position, F1_link, F2_link, accuracy=0.01
):
    contact_index = []
    contact_list = []
    contact_normal_list = []
    contact_local_tangent_list = []
    if F1_link.distance(Polygon(current_vertex_position)) < accuracy:
        if Point(F1_position).distance(Polygon(current_vertex_position)) < accuracy:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and Point(F1_position).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F1_position,
                        )
                    )
                    contact_index.append("F1")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F1_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                F1_position[0] + F1_normal[0] * 0.011,
                                F1_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        else:
            for i in range(len(current_vertex_position)):
                if (
                    "F1" not in contact_index
                    and F1_link.distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F1_contact_position = current_vertex_position[i]
                    contact_list.append(F1_contact_position)
                    contact_index.append("F1")
                    F1_normal = generate_normal_vector(
                        F1_link.coords[:][0], F1_link.coords[:][1]
                    )
                    F1_normal = [-F1_normal[1], F1_normal[0]]
                    if (
                        Point(
                            [
                                F1_contact_position[0] + F1_normal[0] * 0.011,
                                F1_contact_position[1] + F1_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F1_normal = [-F1_normal[0], -F1_normal[1]]
                    contact_normal_list.append(F1_normal)
                    contact_local_tangent_list.append(
                        [list(F1_link.coords[:][0]), list(F1_link.coords[:][1])]
                    )
                    break
    if F2_link.distance(Polygon(current_vertex_position)) < accuracy:
        if Point(F2_position).distance(Polygon(current_vertex_position)) < accuracy:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and Point(F2_position).distance(
                        LineString(
                            [
                                current_vertex_position[i],
                                current_vertex_position[
                                    (i + 1) % (len(current_vertex_position))
                                ],
                            ]
                        )
                    )
                    < accuracy
                ):
                    contact_list.append(
                        footCtoAB(
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                            F2_position,
                        )
                    )
                    contact_index.append("F2")
                    contact_edge_length = LineString(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    ).length
                    F2_normal = [
                        (
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][1]
                            - current_vertex_position[i][1]
                        )
                        / contact_edge_length,
                        -(
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ][0]
                            - current_vertex_position[i][0]
                        )
                        / contact_edge_length,
                    ]
                    if (
                        Point(
                            [
                                F2_position[0] + F2_normal[0] * 0.011,
                                F2_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [
                            current_vertex_position[i],
                            current_vertex_position[
                                (i + 1) % (len(current_vertex_position))
                            ],
                        ]
                    )
                    break
        else:
            for i in range(len(current_vertex_position)):
                if (
                    "F2" not in contact_index
                    and F2_link.distance(Point(current_vertex_position[i])) < accuracy
                ):
                    F2_contact_position = current_vertex_position[i]
                    contact_list.append(F2_contact_position)
                    contact_index.append("F2")
                    F2_normal = generate_normal_vector(
                        F2_link.coords[:][0], F2_link.coords[:][1]
                    )
                    F2_normal = [-F2_normal[1], F2_normal[0]]
                    if (
                        Point(
                            [
                                F2_contact_position[0] + F2_normal[0] * 0.011,
                                F2_contact_position[1] + F2_normal[1] * 0.011,
                            ]
                        ).distance(Polygon(current_vertex_position))
                        != 0
                    ):
                        F2_normal = [-F2_normal[0], -F2_normal[1]]
                    contact_normal_list.append(F2_normal)
                    contact_local_tangent_list.append(
                        [list(F2_link.coords[:][0]), list(F2_link.coords[:][1])]
                    )
                    break
    return contact_index, contact_list, contact_normal_list, contact_local_tangent_list


def circumference(current_vertex_position):
    accumulate_length = LinearRing(current_vertex_position).length
    return accumulate_length


def finger_position2index(current_vertex_position, finger_position, accuracy=0.01):
    finger_position_index = LinearRing(current_vertex_position).project(
        Point(finger_position)
    )
    return finger_position_index


def finger_index2position(current_vertex_position, current_position_index):
    accumulate_length = 0
    finger_position = list(
        list(
            LinearRing(current_vertex_position)
            .interpolate(current_position_index)
            .coords
        )[0]
    )
    return finger_position


def friction_tangent_direction(contact_normal, contact_mode, contact_local_tangent):
    if len(contact_local_tangent) == 2:
        if contact_mode == "S+":
            if (
                np.cross(
                    [-contact_normal[0], -contact_normal[1]],
                    generate_vector(contact_local_tangent[0], contact_local_tangent[1]),
                )
                > 0
            ):
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[0], contact_local_tangent[1]
                )
            else:
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[1], contact_local_tangent[0]
                )
        else:
            if (
                np.cross(
                    [-contact_normal[0], -contact_normal[1]],
                    generate_vector(contact_local_tangent[0], contact_local_tangent[1]),
                )
                < 0
            ):
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[0], contact_local_tangent[1]
                )
            else:
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[1], contact_local_tangent[0]
                )
    else:
        if contact_mode == "S+":
            if (
                np.cross(
                    [-contact_normal[0][0], -contact_normal[0][1]],
                    generate_vector(contact_local_tangent[0], contact_local_tangent[1]),
                )
                > 0
            ):
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[0], contact_local_tangent[1]
                )
            else:
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[2], contact_local_tangent[1]
                )
        else:
            if (
                np.cross(
                    [-contact_normal[0][0], -contact_normal[0][1]],
                    generate_vector(contact_local_tangent[0], contact_local_tangent[1]),
                )
                < 0
            ):
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[0], contact_local_tangent[1]
                )
            else:
                friction_tangent_direction = generate_vector(
                    contact_local_tangent[2], contact_local_tangent[1]
                )
    friction_tangent_direction_length = sqrt(
        friction_tangent_direction[0] ** 2 + friction_tangent_direction[1] ** 2
    )
    friction_tangent_direction = [
        friction_tangent_direction[0] / friction_tangent_direction_length,
        friction_tangent_direction[1] / friction_tangent_direction_length,
    ]
    return friction_tangent_direction


def whether_stable(
    contact_index,
    contact_list,
    contact_normal_list,
    contact_local_tangent_list,
    current_CoM,
    contact_mode,
    mu_gripper=0.6,
    mu_environment=0.03,
):
    # print contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode
    mu_gripper = gol.get_value("mu_gripper", 0.6)
    mu_environment = gol.get_value("mu_environment", 0.03)
    IsStable = False
    contact_normal_and_gravity = np.expand_dims(
        [np.cross(current_CoM, [0, -1]), 0, -1], axis=1
    )
    contact_point_R_column_number = dict()
    column_number = 0
    angle_to_normal_rad = dict()
    for i in range(len(contact_index)):
        if re.match("E", contact_index[i]) != None:
            mu = mu_environment
        else:
            mu = mu_gripper
        if contact_index[i] in contact_mode and (contact_mode[contact_index[i]] == "R"):
            contact_point_R_column_number[contact_index[i]] = []
            if len(np.array(contact_normal_list[i]).shape) == 1:
                fri_cone1 = np.dot(
                    [
                        [cos(atan(-mu)), -sin(atan(-mu))],
                        [sin(atan(-mu)), cos(atan(-mu))],
                    ],
                    np.expand_dims(contact_normal_list[i], axis=1),
                ).transpose()[0]
                fri_cone2 = np.dot(
                    [[cos(atan(mu)), -sin(atan(mu))], [sin(atan(mu)), cos(atan(mu))]],
                    np.expand_dims(contact_normal_list[i], axis=1),
                ).transpose()[0]
                fri_wrench1 = np.expand_dims(
                    [np.cross(contact_list[i], fri_cone1), fri_cone1[0], fri_cone1[1]],
                    axis=1,
                )
                fri_wrench2 = np.expand_dims(
                    [np.cross(contact_list[i], fri_cone2), fri_cone2[0], fri_cone2[1]],
                    axis=1,
                )
                contact_normal_and_gravity = np.hstack(
                    (contact_normal_and_gravity, fri_wrench1, fri_wrench2)
                )
                column_number += 2
                contact_point_R_column_number[contact_index[i]] += [
                    column_number - 1,
                    column_number,
                ]
            else:
                for contact_normal in contact_normal_list[i]:
                    fri_cone1 = np.dot(
                        [
                            [cos(atan(-mu)), -sin(atan(-mu))],
                            [sin(atan(-mu)), cos(atan(-mu))],
                        ],
                        np.expand_dims(contact_normal, axis=1),
                    ).transpose()[0]
                    fri_cone2 = np.dot(
                        [
                            [cos(atan(mu)), -sin(atan(mu))],
                            [sin(atan(mu)), cos(atan(mu))],
                        ],
                        np.expand_dims(contact_normal, axis=1),
                    ).transpose()[0]
                    fri_wrench1 = np.expand_dims(
                        [
                            np.cross(contact_list[i], fri_cone1),
                            fri_cone1[0],
                            fri_cone1[1],
                        ],
                        axis=1,
                    )
                    fri_wrench2 = np.expand_dims(
                        [
                            np.cross(contact_list[i], fri_cone2),
                            fri_cone2[0],
                            fri_cone2[1],
                        ],
                        axis=1,
                    )
                    contact_normal_and_gravity = np.hstack(
                        (contact_normal_and_gravity, fri_wrench1, fri_wrench2)
                    )
                    column_number += 2
                    contact_point_R_column_number[contact_index[i]] += [
                        column_number - 1,
                        column_number,
                    ]
        elif (
            contact_index[i] in contact_mode and contact_mode[contact_index[i]] == "S+"
        ) or (
            contact_index[i] in contact_mode and contact_mode[contact_index[i]] == "S-"
        ):
            fri_cone_list = []
            friction_tangent = friction_tangent_direction(
                contact_normal_list[i],
                contact_mode[contact_index[i]],
                contact_local_tangent_list[i],
            )
            if len(np.array(contact_normal_list[i]).shape) == 1:
                fri_cone_list.append(
                    np.dot(
                        [
                            [cos(atan(-mu)), -sin(atan(-mu))],
                            [sin(atan(-mu)), cos(atan(-mu))],
                        ],
                        np.expand_dims(contact_normal_list[i], axis=1),
                    ).transpose()[0]
                )
                fri_cone_list.append(
                    np.dot(
                        [
                            [cos(atan(mu)), -sin(atan(mu))],
                            [sin(atan(mu)), cos(atan(mu))],
                        ],
                        np.expand_dims(contact_normal_list[i], axis=1),
                    ).transpose()[0]
                )
            else:
                for contact_normal in contact_normal_list[i]:
                    fri_cone_list.append(
                        np.dot(
                            [
                                [cos(atan(-mu)), -sin(atan(-mu))],
                                [sin(atan(-mu)), cos(atan(-mu))],
                            ],
                            np.expand_dims(contact_normal, axis=1),
                        ).transpose()[0]
                    )
                    fri_cone_list.append(
                        np.dot(
                            [
                                [cos(atan(mu)), -sin(atan(mu))],
                                [sin(atan(mu)), cos(atan(mu))],
                            ],
                            np.expand_dims(contact_normal, axis=1),
                        ).transpose()[0]
                    )
            for fri_cone in fri_cone_list:
                if (
                    np.dot(fri_cone, friction_tangent) > 0
                    and abs(
                        np.dot(fri_cone, friction_tangent)
                        / sqrt(1 - (np.dot(fri_cone, friction_tangent)) ** 2)
                        - mu
                    )
                    < 0.001
                ):
                    fri_wrench = np.expand_dims(
                        [np.cross(contact_list[i], fri_cone), fri_cone[0], fri_cone[1]],
                        axis=1,
                    )
                    contact_normal_and_gravity = np.hstack(
                        (contact_normal_and_gravity, fri_wrench)
                    )
                    column_number += 1
                    break
    if np.linalg.matrix_rank(contact_normal_and_gravity) == 3:
        f = [0] * contact_normal_and_gravity.shape[1]
        MODEL = gurobipy.Model("linear_program")
        MODEL.params.LogToConsole = 0
        x = MODEL.addVars(contact_normal_and_gravity.shape[1], lb=1, name="x")
        MODEL.update()
        MODEL.setObjective(x.prod(f), gurobipy.GRB.MINIMIZE)
        MODEL.addConstrs(
            x.prod(contact_normal_and_gravity.tolist()[i]) == 0
            for i in range(contact_normal_and_gravity.shape[0])
        )
        MODEL.optimize()
        try:
            MODEL.objVal
            IsStable = True
            contact_normal_and_gravity_no_torque = np.delete(
                contact_normal_and_gravity, 0, axis=0
            )
            for (
                point_index,
                point_index_column_number,
            ) in contact_point_R_column_number.items():
                resultant = np.array([[0.0, 0.0]]).transpose()
                for number in point_index_column_number:
                    resultant += (
                        np.expand_dims(
                            contact_normal_and_gravity_no_torque[:, number], axis=1
                        )
                        * [
                            MODEL.getVars()[iff].x
                            for iff in range(len(MODEL.getVars()))
                        ][number]
                    )
                resultant = resultant.transpose()[0].tolist()
                if (
                    len(
                        np.array(
                            contact_normal_list[contact_index.index(point_index)]
                        ).shape
                    )
                    == 1
                ):
                    average_contact_normal = contact_normal_list[
                        contact_index.index(point_index)
                    ]
                else:
                    average_contact_normal = np.array([[0.0, 0.0]])
                    for contact_normal in contact_normal_list[i]:
                        average_contact_normal += np.array([contact_normal])
                    average_contact_normal = average_contact_normal[0].tolist()
                angle_to_normal_rad[point_index] = acos(
                    max(
                        min(
                            np.dot(average_contact_normal, resultant)
                            / (
                                sqrt(
                                    average_contact_normal[0] ** 2
                                    + average_contact_normal[1] ** 2
                                )
                                * sqrt(resultant[0] ** 2 + resultant[1] ** 2)
                            ),
                            1,
                        ),
                        -1,
                    )
                )
        except:
            return False, dict()
    return IsStable, angle_to_normal_rad


def whether_force_closure(current_vertex_position, F1, F2, mu_gripper=0.6):
    mu_gripper = gol.get_value("mu_gripper", 0.6)
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = (
        contact_points_and_normal_finger(current_vertex_position, F1, F2, accuracy=0.01)
    )
    F1_index = contact_index.index("F1")
    F2_index = contact_index.index("F2")
    contact_local_tangent_F1 = (
        np.array(contact_local_tangent_list[F1_index][1])
        - np.array(contact_local_tangent_list[F1_index][0])
        / LineString(contact_local_tangent_list[F1_index]).length
    )
    contact_local_tangent_F2 = (
        np.array(contact_local_tangent_list[F2_index][1])
        - np.array(contact_local_tangent_list[F2_index][0])
        / LineString(contact_local_tangent_list[F2_index]).length
    )
    if (
        np.cross(
            np.array(F2) - np.array(F1),
            np.array(contact_normal_list[F1_index])
            - mu_gripper * contact_local_tangent_F1,
        )
        >= 0
        and np.cross(
            np.array(F2) - np.array(F1),
            np.array(contact_normal_list[F1_index])
            + mu_gripper * contact_local_tangent_F1,
        )
        <= 0
        and np.cross(
            np.array(F1) - np.array(F2),
            np.array(contact_normal_list[F2_index])
            - mu_gripper * contact_local_tangent_F2,
        )
        >= 0
        and np.cross(
            np.array(F1) - np.array(F2),
            np.array(contact_normal_list[F2_index])
            + mu_gripper * contact_local_tangent_F2,
        )
        <= 0
    ):
        return True
    else:
        return False

def whether_stable_simple(
    contact_index,
    contact_list,
    contact_normal_list,
    contact_local_tangent_list,
    current_CoM,
    contact_mode,
    mu_gripper=0.6,
    mu_environment=0.03,
):
    mu_gripper = gol.get_value("mu_gripper", 0.6)
    mu_environment = gol.get_value("mu_environment", 0.03)
    IsStable = False
    contact_normal_and_gravity = np.expand_dims(
        [np.cross(np.array(current_CoM), np.array([0, -1])), 0, -1], axis=1
    )
    # time3 = time.time()
    for i in range(len(contact_index)):
        if re.match("E", contact_index[i]) != None:
            mu = mu_environment
        else:
            mu = mu_gripper
        if contact_index[i] in contact_mode and (contact_mode[contact_index[i]] == "R"):
            if len(np.array(contact_normal_list[i]).shape) == 1:
                fri_cone1 = np.dot(
                    np.array(
                        [
                            [cos(atan(-mu)), -sin(atan(-mu))],
                            [sin(atan(-mu)), cos(atan(-mu))],
                        ]
                    ),
                    np.expand_dims(np.array(contact_normal_list[i]), axis=1),
                ).transpose()[0]
                fri_cone2 = np.dot(
                    np.array(
                        [
                            [cos(atan(mu)), -sin(atan(mu))],
                            [sin(atan(mu)), cos(atan(mu))],
                        ]
                    ),
                    np.expand_dims(np.array(contact_normal_list[i]), axis=1),
                ).transpose()[0]
                fri_wrench1 = np.expand_dims(
                    np.array(
                        [
                            np.cross(contact_list[i], fri_cone1).item(),
                            fri_cone1[0].item(),
                            fri_cone1[1].item(),
                        ]
                    ),
                    axis=1,
                )
                fri_wrench2 = np.expand_dims(
                    np.array(
                        [
                            np.cross(contact_list[i], fri_cone2).item(),
                            fri_cone2[0].item(),
                            fri_cone2[1].item(),
                        ]
                    ),
                    axis=1,
                )
                contact_normal_and_gravity = np.hstack(
                    (contact_normal_and_gravity, fri_wrench1, fri_wrench2)
                )
            else:
                for contact_normal in contact_normal_list[i]:
                    fri_cone1 = np.dot(
                        [
                            [cos(atan(-mu)), -sin(atan(-mu))],
                            [sin(atan(-mu)), cos(atan(-mu))],
                        ],
                        np.expand_dims(contact_normal, axis=1),
                    ).transpose()[0]
                    fri_cone2 = np.dot(
                        [
                            [cos(atan(mu)), -sin(atan(mu))],
                            [sin(atan(mu)), cos(atan(mu))],
                        ],
                        np.expand_dims(contact_normal, axis=1),
                    ).transpose()[0]
                    fri_wrench1 = np.expand_dims(
                        [
                            np.cross(contact_list[i], fri_cone1),
                            fri_cone1[0],
                            fri_cone1[1],
                        ],
                        axis=1,
                    )
                    fri_wrench2 = np.expand_dims(
                        [
                            np.cross(contact_list[i], fri_cone2),
                            fri_cone2[0],
                            fri_cone2[1],
                        ],
                        axis=1,
                    )
                    contact_normal_and_gravity = np.hstack(
                        (contact_normal_and_gravity, fri_wrench1, fri_wrench2)
                    )
        elif (
            contact_index[i] in contact_mode and contact_mode[contact_index[i]] == "S+"
        ) or (
            contact_index[i] in contact_mode and contact_mode[contact_index[i]] == "S-"
        ):
            fri_cone_list = []
            friction_tangent = friction_tangent_direction(
                contact_normal_list[i],
                contact_mode[contact_index[i]],
                contact_local_tangent_list[i],
            )
            if len(np.array(contact_normal_list[i]).shape) == 1:
                fri_cone_list.append(
                    np.dot(
                        [
                            [cos(atan(-mu)), -sin(atan(-mu))],
                            [sin(atan(-mu)), cos(atan(-mu))],
                        ],
                        np.expand_dims(contact_normal_list[i], axis=1),
                    ).transpose()[0]
                )
                fri_cone_list.append(
                    np.dot(
                        [
                            [cos(atan(mu)), -sin(atan(mu))],
                            [sin(atan(mu)), cos(atan(mu))],
                        ],
                        np.expand_dims(contact_normal_list[i], axis=1),
                    ).transpose()[0]
                )
            else:
                for contact_normal in contact_normal_list[i]:
                    fri_cone_list.append(
                        np.dot(
                            [
                                [cos(atan(-mu)), -sin(atan(-mu))],
                                [sin(atan(-mu)), cos(atan(-mu))],
                            ],
                            np.expand_dims(contact_normal, axis=1),
                        ).transpose()[0]
                    )
                    fri_cone_list.append(
                        np.dot(
                            [
                                [cos(atan(mu)), -sin(atan(mu))],
                                [sin(atan(mu)), cos(atan(mu))],
                            ],
                            np.expand_dims(contact_normal, axis=1),
                        ).transpose()[0]
                    )
            for fri_cone in fri_cone_list:
                if (
                    np.dot(fri_cone, friction_tangent) > 0
                    and abs(
                        np.dot(fri_cone, friction_tangent)
                        / sqrt(1 - (np.dot(fri_cone, friction_tangent)) ** 2)
                        - mu
                    )
                    < 0.001
                ):
                    fri_wrench = np.expand_dims(
                        [np.cross(contact_list[i], fri_cone), fri_cone[0], fri_cone[1]],
                        axis=1,
                    )
                    contact_normal_and_gravity = np.hstack(
                        (contact_normal_and_gravity, fri_wrench)
                    )
                    break
    # print time.time()-time3
    if np.linalg.matrix_rank(contact_normal_and_gravity) == 3:
        f = [0] * contact_normal_and_gravity.shape[1]
        MODEL = gurobipy.Model("linear_program")
        MODEL.params.LogToConsole = 0
        x = MODEL.addVars(contact_normal_and_gravity.shape[1], lb=1, name="x")
        MODEL.update()
        MODEL.setObjective(x.prod(f), gurobipy.GRB.MINIMIZE)
        MODEL.addConstrs(
            x.prod(contact_normal_and_gravity.tolist()[i]) == 0
            for i in range(contact_normal_and_gravity.shape[0])
        )
        MODEL.optimize()
        try:
            MODEL.objVal
            IsStable = True
        except:
            IsStable = False
    return IsStable
