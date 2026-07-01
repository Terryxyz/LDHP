#!/usr/bin/env python
import random
import re
import time
from math import *

import numpy as np
from general_functions import *
from jerk import *
from shapely.geometry import LineString, Point, Polygon


def obtain_contact_state(
    current_vertex_position,
    profile_environment,
    current_vertex_position_buffer=None,
    accuracy=0.004,
):
    V = current_vertex_position
    l = [
        [V[i], V[(i + 1) % (len(current_vertex_position))]]
        for i in range(len(current_vertex_position))
    ]
    E = dict()
    for i in range(len(profile_environment) - 1):
        E[i + 1] = [
            profile_environment[len(profile_environment) - i - 2],
            profile_environment[len(profile_environment) - i - 1],
        ]
    contact_state_list = []
    if current_vertex_position_buffer == None:
        if (
            Polygon(current_vertex_position)
            .buffer(-accuracy)
            .distance(LineString(profile_environment))
            == 0
        ):
            return ["Invalid"]
    else:
        if (
            Polygon(current_vertex_position_buffer).distance(
                LineString(profile_environment)
            )
            == 0
        ):
            return ["Invalid"]
    if (
        Polygon(current_vertex_position).distance(LineString(profile_environment))
        > accuracy * 2
    ):
        return ["no_contact"]
    for i in range(len(current_vertex_position)):
        for E_key in E:
            # print abs(np.cross(generate_vector(V[i], V[(i+1)%(len(current_vertex_position))]), generate_vector(E[E_key][0], E[E_key][1]))/(LineString([V[i], V[(i+1)%(len(current_vertex_position))]]).length*LineString(E[E_key]).length))
            if (
                abs(
                    np.cross(
                        generate_vector(
                            V[i], V[(i + 1) % (len(current_vertex_position))]
                        ),
                        generate_vector(E[E_key][0], E[E_key][1]),
                    )
                    / (
                        LineString(
                            [V[i], V[(i + 1) % (len(current_vertex_position))]]
                        ).length
                        * LineString(E[E_key]).length
                    )
                )
                > 0.017
            ):
                continue
            if (
                Point(V[i]).distance(LineString(E[E_key])) < accuracy
                and Point(V[(i + 1) % (len(current_vertex_position))]).distance(
                    LineString(E[E_key])
                )
                < accuracy
            ):
                contact_state_list.append("l" + str(i + 1) + "E" + str(E_key))
            else:
                temp_point0 = [
                    V[i][0]
                    + (V[(i + 1) % (len(current_vertex_position))][0] - V[i][0])
                    / 10
                    * 3,
                    V[i][1]
                    + (V[(i + 1) % (len(current_vertex_position))][1] - V[i][1])
                    / 10
                    * 2,
                ]
                if (
                    Point(V[i]).distance(LineString(E[E_key])) < accuracy
                    and Point(temp_point0).distance(LineString(E[E_key])) < accuracy
                ):
                    contact_state_list.append("l" + str(i + 1) + "E" + str(E_key))
                else:
                    temp_point1 = [
                        V[i][0]
                        + (V[(i + 1) % (len(current_vertex_position))][0] - V[i][0])
                        / 10
                        * 8,
                        V[i][1]
                        + (V[(i + 1) % (len(current_vertex_position))][1] - V[i][1])
                        / 10
                        * 8,
                    ]
                    if (
                        Point(V[(i + 1) % (len(current_vertex_position))]).distance(
                            LineString(E[E_key])
                        )
                        < accuracy
                        and Point(temp_point1).distance(LineString(E[E_key])) < accuracy
                    ):
                        contact_state_list.append("l" + str(i + 1) + "E" + str(E_key))
    for i in range(len(current_vertex_position)):
        for E_key in E:
            if ("l" + str(i + 1) + "E" + str(E_key)) not in contact_state_list and (
                "l"
                + str((i - 1) % (len(current_vertex_position)) + 1)
                + "E"
                + str(E_key)
            ) not in contact_state_list:
                if Point(V[i]).distance(LineString(E[E_key])) < accuracy:
                    contact_state_list.append("V" + str(i + 1) + "E" + str(E_key))
    if len(contact_state_list) == 1:
        contact_state_list.append("single_contact")
    elif len(contact_state_list) == 2:
        contact_state_list.append("double_contact")
    else:
        return ["Invalid"]
    return sorted(contact_state_list)


# print obtain_contact_state([[0.5,0], [0.4, 0.3], [0.5,0], [0.7,0.3], [1.2,0], [1.2, 1.5], [0.5, 1.5]], [[0,0], [3,0]] )


def whether_contains(
    principle_contact_state1, principle_contact_state2, object_edge_number
):
    # print principle_contact_state1, principle_contact_state2
    if principle_contact_state1 == principle_contact_state2:
        return False
    elif (
        principle_contact_state1[principle_contact_state1.index("E") : :]
        != principle_contact_state2[principle_contact_state2.index("E") : :]
    ):
        return False
    elif principle_contact_state1[0] == "V" and principle_contact_state2[0] == "l":
        return False
    elif principle_contact_state1[0] == principle_contact_state2[0]:
        return False
    elif (
        int(principle_contact_state1[1 : principle_contact_state1.index("E")])
        % object_edge_number
        + 1
    ) == int(principle_contact_state2[1 : principle_contact_state2.index("E")]):
        return True
    elif int(principle_contact_state1[1 : principle_contact_state1.index("E")]) == int(
        principle_contact_state2[1 : principle_contact_state2.index("E")]
    ):
        return True
    else:
        return False


# print whether_contains('l4E2', 'V4E2', 4)


def whether_less_constraint(contact_state1, contact_state2, object_edge_number):
    temp_contact_state1 = contact_state1[:]
    temp_contact_state2 = contact_state2[:]
    if sorted(contact_state1) == sorted(contact_state2):
        return False
    if "single_contact" in temp_contact_state1:
        temp_contact_state1.remove("single_contact")
    if "single_contact" in temp_contact_state2:
        temp_contact_state2.remove("single_contact")
    if "double_contact" in temp_contact_state1:
        temp_contact_state1.remove("double_contact")
    if "double_contact" in temp_contact_state2:
        temp_contact_state2.remove("double_contact")
    if len(temp_contact_state1) < len(temp_contact_state2):
        return False
    satisfying_principle_contact_number = 0
    contain_list = dict()
    for principle_contact in temp_contact_state2:
        if principle_contact in temp_contact_state1:
            satisfying_principle_contact_number += 1
            continue
        else:
            contain_list[principle_contact] = []
            for principle_contact_state_1 in temp_contact_state1:
                if whether_contains(
                    principle_contact_state_1, principle_contact, object_edge_number
                ):
                    contain_list[principle_contact].append(principle_contact_state_1)
            if len(contain_list[principle_contact]) == 1:
                satisfying_principle_contact_number += 1
    if satisfying_principle_contact_number == len(temp_contact_state2):
        for principle_contact in temp_contact_state2:
            for principle_contact0 in temp_contact_state2:
                if principle_contact0 == principle_contact:
                    continue
                if (
                    principle_contact in contain_list
                    and principle_contact0 in contain_list
                    and set(contain_list[principle_contact]).intersection(
                        set(contain_list[principle_contact0])
                    )
                    != set()
                ):
                    return False
        return True
    return False


# print whether_less_constraint(['double_contact', 'l3E2', 'l4E1'], ['V1E1', 'V4E2', 'double_contact'], 4)


def finding_configuration_cos_contact_state(
    current_object_vertex,
    environment,
    target_contact_state,
    scope=[[-0.1, -0.1], [3.1, -0.1], [3.1, 3.1], [-0.1, 3.1]],
):
    # print 'aaa', target_contact_state
    if target_contact_state == ["Invalid"] or target_contact_state == ["no_contact"]:
        return False
    elif "single_contact" in target_contact_state:
        temp_target_contact_state = target_contact_state[:]
        temp_target_contact_state.remove("single_contact")
        temp_target_contact_state = temp_target_contact_state[0]
        if temp_target_contact_state[0] == "l":
            mating_edge_index = int(
                temp_target_contact_state[1 : temp_target_contact_state.index("E")]
            )
            mating_environment_index = int(
                temp_target_contact_state[temp_target_contact_state.index("E") + 1 : :]
            )
            mating_edge = [
                current_object_vertex[mating_edge_index - 1],
                current_object_vertex[mating_edge_index % len(current_object_vertex)],
            ]
            mating_environment_edge = [
                environment[len(environment) - mating_environment_index - 1],
                environment[len(environment) - mating_environment_index],
            ]
            rotate_angle = calculate_intersect_angle_AB_CD(
                mating_edge[0],
                mating_edge[1],
                mating_environment_edge[0],
                mating_environment_edge[1],
            )
            object_vertex_position_after_rotation = [
                point_position_after_rotation(
                    vertex, current_object_vertex[0], rotate_angle
                )
                for vertex in current_object_vertex
            ]
            mating_edge = [
                object_vertex_position_after_rotation[mating_edge_index - 1],
                object_vertex_position_after_rotation[
                    mating_edge_index % len(current_object_vertex)
                ],
            ]
            translating_distance = abs(
                CtoAB(
                    mating_environment_edge[0],
                    mating_environment_edge[1],
                    mating_edge[0],
                )
            )
            translating_direction = generate_vector(mating_edge[0], mating_edge[1])
            translating_direction = [
                translating_direction[1] / LineString(mating_edge).length,
                translating_direction[0] / LineString(mating_edge).length,
            ]
            if (
                np.dot(
                    translating_direction,
                    generate_vector(mating_environment_edge[0], mating_edge[0]),
                )
                > 0
            ):
                translating_direction = [
                    -translating_direction[0],
                    -translating_direction[1],
                ]
            object_vertex_position_after_translation = [
                [
                    vertex[0] + translating_distance * translating_direction[0],
                    vertex[1] + translating_distance * translating_direction[1],
                ]
                for vertex in object_vertex_position_after_rotation
            ]
            if sorted(
                obtain_contact_state(
                    object_vertex_position_after_translation, environment
                )
            ) == sorted(target_contact_state):
                if Polygon(object_vertex_position_after_translation).within(
                    Polygon(scope)
                ):
                    return object_vertex_position_after_translation
            mating_edge = [
                object_vertex_position_after_translation[mating_edge_index - 1],
                object_vertex_position_after_translation[
                    mating_edge_index % len(current_object_vertex)
                ],
            ]
            translating_distance = LineString(mating_edge).distance(
                LineString(mating_environment_edge)
            )
            mating_edge_direction = generate_vector(mating_edge[0], mating_edge[1])
            translating_direction = [
                mating_edge_direction[0] / LineString(mating_edge).length,
                mating_edge_direction[1] / LineString(mating_edge).length,
            ]
            object_vertex_position_after_translation = [
                [
                    vertex[0] + translating_distance * translating_direction[0],
                    vertex[1] + translating_distance * translating_direction[1],
                ]
                for vertex in object_vertex_position_after_translation
            ]
            mating_edge = [
                object_vertex_position_after_translation[mating_edge_index - 1],
                object_vertex_position_after_translation[
                    mating_edge_index % len(current_object_vertex)
                ],
            ]
            if (
                LineString(mating_edge).distance(LineString(mating_environment_edge))
                != 0
            ):
                translating_direction = [
                    -translating_direction[0],
                    -translating_direction[1],
                ]
                object_vertex_position_after_translation = [
                    [
                        vertex[0] + 2 * translating_distance * translating_direction[0],
                        vertex[1] + 2 * translating_distance * translating_direction[1],
                    ]
                    for vertex in object_vertex_position_after_translation
                ]
            object_vertex_position_before_translation = (
                object_vertex_position_after_translation[:]
            )
            mating_edge = [
                object_vertex_position_before_translation[mating_edge_index - 1],
                object_vertex_position_before_translation[
                    mating_edge_index % len(current_object_vertex)
                ],
            ]
            for translation_step in np.append(
                np.arange(
                    0,
                    -LineString(mating_edge).length
                    - LineString(mating_environment_edge).length,
                    -(
                        LineString(mating_edge).length
                        + LineString(mating_environment_edge).length
                    )
                    / 40,
                ),
                np.arange(
                    0,
                    LineString(mating_edge).length
                    + LineString(mating_environment_edge).length,
                    (
                        LineString(mating_edge).length
                        + LineString(mating_environment_edge).length
                    )
                    / 40,
                ),
            ):
                object_vertex_position_after_translation = [
                    [
                        vertex[0] + translation_step * translating_direction[0],
                        vertex[1] + translation_step * translating_direction[1],
                    ]
                    for vertex in object_vertex_position_before_translation
                ]
                if sorted(
                    obtain_contact_state(
                        object_vertex_position_after_translation, environment
                    )
                ) == sorted(target_contact_state):
                    if Polygon(object_vertex_position_after_translation).within(
                        Polygon(scope)
                    ):
                        return object_vertex_position_after_translation
            else:
                object_vertex_position_before_translation = [
                    point_position_after_rotation(vertex, mating_edge[0], 180)
                    for vertex in object_vertex_position_before_translation
                ]
                for translation_step in np.append(
                    np.arange(
                        0,
                        -LineString(mating_edge).length
                        - LineString(mating_environment_edge).length,
                        -(
                            LineString(mating_edge).length
                            + LineString(mating_environment_edge).length
                        )
                        / 15,
                    ),
                    np.arange(
                        0,
                        LineString(mating_edge).length
                        + LineString(mating_environment_edge).length,
                        (
                            LineString(mating_edge).length
                            + LineString(mating_environment_edge).length
                        )
                        / 15,
                    ),
                ):
                    object_vertex_position_after_translation = [
                        [
                            vertex[0] + translation_step * translating_direction[0],
                            vertex[1] + translation_step * translating_direction[1],
                        ]
                        for vertex in object_vertex_position_before_translation
                    ]
                    if sorted(
                        obtain_contact_state(
                            object_vertex_position_after_translation, environment
                        )
                    ) == sorted(target_contact_state):
                        if Polygon(object_vertex_position_after_translation).within(
                            Polygon(scope)
                        ):
                            return object_vertex_position_after_translation
        if temp_target_contact_state[0] == "V":
            mating_vertex_index = int(
                temp_target_contact_state[1 : temp_target_contact_state.index("E")]
            )
            mating_environment_index = int(
                temp_target_contact_state[temp_target_contact_state.index("E") + 1 : :]
            )
            mating_vertex = current_object_vertex[mating_vertex_index - 1]
            mating_environment_edge = [
                environment[len(environment) - mating_environment_index - 1],
                environment[len(environment) - mating_environment_index],
            ]
            mating_environment_edge_midpoint = [
                (mating_environment_edge[0][0] + mating_environment_edge[1][0]) / 2.0,
                (mating_environment_edge[0][1] + mating_environment_edge[1][1]) / 2.0,
            ]
            translatig_direction_and_distance = generate_vector(
                mating_vertex, mating_environment_edge_midpoint
            )
            object_vertex_position_after_translation = [
                [
                    vertex[0] + translatig_direction_and_distance[0],
                    vertex[1] + translatig_direction_and_distance[1],
                ]
                for vertex in current_object_vertex
            ]
            mating_vertex = object_vertex_position_after_translation[
                mating_vertex_index - 1
            ]
            for rotation_step in np.append(
                np.arange(0, 180, 5), np.arange(0, -180, -5)
            ):
                object_vertex_position_after_rotation = [
                    point_position_after_rotation(vertex, mating_vertex, rotation_step)
                    for vertex in object_vertex_position_after_translation
                ]
                if sorted(
                    obtain_contact_state(
                        object_vertex_position_after_rotation, environment
                    )
                ) == sorted(target_contact_state):
                    if Polygon(object_vertex_position_after_rotation).within(
                        Polygon(scope)
                    ):
                        return object_vertex_position_after_rotation
    else:
        temp_target_contact_state = target_contact_state[:]
        temp_target_contact_state.remove("double_contact")
        temp_target_contact_state0 = temp_target_contact_state[0]
        temp_target_contact_state1 = temp_target_contact_state[1]
        if temp_target_contact_state0 == temp_target_contact_state1:
            return False
        elif whether_contains(
            temp_target_contact_state0,
            temp_target_contact_state1,
            len(current_object_vertex),
        ) or whether_contains(
            temp_target_contact_state1,
            temp_target_contact_state0,
            len(current_object_vertex),
        ):
            return False
        if (
            temp_target_contact_state0[temp_target_contact_state0.index("E") + 1 : :]
            == temp_target_contact_state1[temp_target_contact_state1.index("E") + 1 : :]
        ):
            if (
                (
                    temp_target_contact_state0[0] == "V"
                    and temp_target_contact_state1[0] == "V"
                )
                or (
                    temp_target_contact_state0[0] == "l"
                    and temp_target_contact_state1[0] == "l"
                )
            ) and (
                int(
                    temp_target_contact_state0[
                        1 : temp_target_contact_state0.index("E")
                    ]
                )
                % len(current_object_vertex)
                + 1
                == int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                or int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                % len(current_object_vertex)
                + 1
                == int(
                    temp_target_contact_state0[
                        1 : temp_target_contact_state0.index("E")
                    ]
                )
            ):
                return False
            if (
                temp_target_contact_state0[0] == "l"
                and temp_target_contact_state1[0] == "l"
            ):
                mating_edge_index0 = int(
                    temp_target_contact_state0[
                        1 : temp_target_contact_state0.index("E")
                    ]
                )
                mating_edge_index1 = int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                mating_edge0 = [
                    current_object_vertex[mating_edge_index0 - 1],
                    current_object_vertex[
                        mating_edge_index0 % len(current_object_vertex)
                    ],
                ]
                mating_edge1 = [
                    current_object_vertex[mating_edge_index1 - 1],
                    current_object_vertex[
                        mating_edge_index1 % len(current_object_vertex)
                    ],
                ]
                mating_edge0_edge1_link = [mating_edge0[1], mating_edge1[0]]
                if (
                    abs(
                        np.cross(
                            generate_vector(mating_edge0[0], mating_edge0[1]),
                            generate_vector(
                                mating_edge0_edge1_link[0], mating_edge0_edge1_link[1]
                            ),
                        )
                    )
                    / (
                        LineString(mating_edge0).length
                        * LineString(mating_edge0_edge1_link).length
                    )
                    > 0.017
                ):
                    return False
                if (
                    abs(
                        np.cross(
                            generate_vector(mating_edge1[0], mating_edge1[1]),
                            generate_vector(
                                mating_edge0_edge1_link[0], mating_edge0_edge1_link[1]
                            ),
                        )
                    )
                    / (
                        LineString(mating_edge1).length
                        * LineString(mating_edge0_edge1_link).length
                    )
                    > 0.017
                ):
                    return False
                if mating_edge_index0 < mating_edge_index1:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_edge_index0 - 1)]
                        + [
                            current_object_vertex[mating_edge_index0 - 1],
                            current_object_vertex[mating_edge_index1],
                        ]
                        + current_object_vertex[(mating_edge_index1 + 1) : :]
                    )
                    mating_index_new = mating_edge_index0
                else:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_edge_index1 - 1)]
                        + [
                            current_object_vertex[mating_edge_index1 - 1],
                            current_object_vertex[mating_edge_index0],
                        ]
                        + current_object_vertex[(mating_edge_index0 + 1) : :]
                    )
                    mating_index_new = mating_edge_index1
            elif (
                temp_target_contact_state0[0] == "V"
                and temp_target_contact_state1[0] == "V"
            ):
                if (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l" + temp_target_contact_state0[1::],
                            temp_target_contact_state1,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                elif (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l"
                            + str(
                                (
                                    int(
                                        temp_target_contact_state0[
                                            1 : temp_target_contact_state0.index("E")
                                        ]
                                    )
                                    - 2
                                )
                                % len(current_object_vertex)
                                + 1
                            )
                            + temp_target_contact_state0[
                                temp_target_contact_state0.index("E") : :
                            ],
                            temp_target_contact_state1,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                elif (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l" + temp_target_contact_state1[1::],
                            temp_target_contact_state0,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                elif (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l"
                            + str(
                                (
                                    int(
                                        temp_target_contact_state1[
                                            1 : temp_target_contact_state1.index("E")
                                        ]
                                    )
                                    - 2
                                )
                                % len(current_object_vertex)
                                + 1
                            )
                            + temp_target_contact_state1[
                                temp_target_contact_state1.index("E") : :
                            ],
                            temp_target_contact_state0,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                mating_vertex_index0 = int(
                    temp_target_contact_state0[
                        1 : temp_target_contact_state0.index("E")
                    ]
                )
                mating_vertex_index1 = int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                mating_vertex0 = current_object_vertex[mating_vertex_index0 - 1]
                mating_vertex1 = current_object_vertex[mating_vertex_index1 - 1]
                if mating_vertex_index0 < mating_vertex_index1:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_vertex_index0 - 1)]
                        + [
                            current_object_vertex[mating_vertex_index0 - 1],
                            current_object_vertex[mating_vertex_index1 - 1],
                        ]
                        + current_object_vertex[mating_vertex_index1::]
                    )
                    mating_index_new = mating_vertex_index0
                else:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_vertex_index1 - 1)]
                        + [
                            current_object_vertex[mating_vertex_index1 - 1],
                            current_object_vertex[mating_vertex_index0 - 1],
                        ]
                        + current_object_vertex[mating_vertex_index0::]
                    )
                    mating_index_new = mating_vertex_index1
            else:
                if (
                    temp_target_contact_state0[0] == "l"
                    and temp_target_contact_state1[0] == "V"
                ):
                    temp = temp_target_contact_state0[:]
                    temp_target_contact_state0 = temp_target_contact_state1[:]
                    temp_target_contact_state1 = temp[:]
                if (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l" + temp_target_contact_state0[1::],
                            temp_target_contact_state1,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                elif (
                    finding_configuration_cos_contact_state(
                        current_object_vertex,
                        environment,
                        [
                            "l"
                            + str(
                                (
                                    int(
                                        temp_target_contact_state0[
                                            1 : temp_target_contact_state0.index("E")
                                        ]
                                    )
                                    - 2
                                )
                                % len(current_object_vertex)
                                + 1
                            )
                            + temp_target_contact_state0[
                                temp_target_contact_state0.index("E") : :
                            ],
                            temp_target_contact_state1,
                            "double_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    return False
                mating_vertex_index0 = int(
                    temp_target_contact_state0[
                        1 : temp_target_contact_state0.index("E")
                    ]
                )
                mating_edge_index1 = int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                mating_vertex0 = current_object_vertex[mating_vertex_index0 - 1]
                mating_edge1 = [
                    current_object_vertex[mating_edge_index1 - 1],
                    current_object_vertex[
                        mating_edge_index1 % len(current_object_vertex)
                    ],
                ]
                mating_vertex0_edge1_link = [mating_vertex0, mating_edge1[0]]
                if (
                    abs(
                        np.cross(
                            generate_vector(mating_edge1[0], mating_edge1[1]),
                            generate_vector(
                                mating_vertex0_edge1_link[0],
                                mating_vertex0_edge1_link[1],
                            ),
                        )
                    )
                    / (
                        LineString(mating_edge1).length
                        * LineString(mating_vertex0_edge1_link).length
                    )
                    > 0.017
                ):
                    return False
                if mating_vertex_index0 < mating_edge_index1:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_vertex_index0 - 1)]
                        + [
                            current_object_vertex[mating_vertex_index0 - 1],
                            current_object_vertex[mating_edge_index1],
                        ]
                        + current_object_vertex[(mating_edge_index1 + 1) : :]
                    )
                    mating_index_new = mating_vertex_index0
                else:
                    bounded_object_vertex = (
                        current_object_vertex[0 : max(0, mating_edge_index1 - 1)]
                        + [
                            current_object_vertex[mating_edge_index1 - 1],
                            current_object_vertex[mating_vertex_index0 - 1],
                        ]
                        + current_object_vertex[(mating_vertex_index0)::]
                    )
                    mating_index_new = mating_edge_index1
            bounded_object_vertex_after_moving = (
                finding_configuration_cos_contact_state(
                    bounded_object_vertex,
                    environment,
                    [
                        "l"
                        + str(mating_index_new)
                        + "E"
                        + temp_target_contact_state0[
                            temp_target_contact_state0.index("E") + 1 : :
                        ],
                        "single_contact",
                    ],
                    scope,
                )
            )
            if not bounded_object_vertex_after_moving:
                return False
            mating_edge_new = [
                bounded_object_vertex_after_moving[mating_index_new - 1],
                bounded_object_vertex_after_moving[mating_index_new],
            ]
            mating_environment_index = int(
                temp_target_contact_state1[
                    temp_target_contact_state1.index("E") + 1 : :
                ]
            )
            mating_environment_edge = [
                environment[len(environment) - mating_environment_index - 1],
                environment[len(environment) - mating_environment_index],
            ]
            mating_environment_edge_direction = [
                (mating_environment_edge[1][0] - mating_environment_edge[0][0])
                / LineString(mating_environment_edge).length,
                (mating_environment_edge[1][1] - mating_environment_edge[0][1])
                / LineString(mating_environment_edge).length,
            ]
            for translating_step in np.append(
                np.arange(
                    0,
                    -LineString(mating_environment_edge).length,
                    -LineString(mating_environment_edge).length / 15.0,
                ),
                np.arange(
                    0,
                    LineString(mating_environment_edge).length,
                    LineString(mating_environment_edge).length / 15.0,
                ),
            ):
                bounded_object_vertex_after_translation = [
                    [
                        vertex[0]
                        + translating_step * mating_environment_edge_direction[0],
                        vertex[1]
                        + translating_step * mating_environment_edge_direction[1],
                    ]
                    for vertex in bounded_object_vertex_after_moving
                ]
                if LineString(
                    [
                        bounded_object_vertex_after_translation[mating_index_new - 1],
                        bounded_object_vertex_after_translation[mating_index_new],
                    ]
                ).within(LineString(mating_environment_edge).buffer(0.005)):
                    delta_translate, _, A_rad = (
                        calculate_parameter_translate_and_rotate_moveA(
                            bounded_object_vertex[0],
                            bounded_object_vertex[1],
                            bounded_object_vertex_after_translation[0],
                            bounded_object_vertex_after_translation[1],
                        )
                    )
                    object_vertex_position_after_translation = [
                        [vertex[0] + delta_translate[0], vertex[1] + delta_translate[1]]
                        for vertex in current_object_vertex
                    ]
                    object_vertex_position_after_rotation = [
                        point_position_after_rotation(
                            vertex,
                            object_vertex_position_after_translation[0],
                            A_rad * 180 / pi,
                        )
                        for vertex in object_vertex_position_after_translation
                    ]
                    if sorted(
                        obtain_contact_state(
                            object_vertex_position_after_rotation, environment
                        )
                    ) == sorted(target_contact_state):
                        return object_vertex_position_after_rotation
        elif (
            temp_target_contact_state0[0] == "l"
            and temp_target_contact_state1[0] == "l"
        ):
            object_vertex_position_after_one_mate = (
                finding_configuration_cos_contact_state(
                    current_object_vertex,
                    environment,
                    [temp_target_contact_state0, "single_contact"],
                    scope,
                )
            )
            if not object_vertex_position_after_one_mate:
                return False
            mating_edge_index = int(
                temp_target_contact_state1[1 : temp_target_contact_state1.index("E")]
            )
            mating_environment_index = int(
                temp_target_contact_state1[
                    temp_target_contact_state1.index("E") + 1 : :
                ]
            )
            mating_edge = [
                object_vertex_position_after_one_mate[mating_edge_index - 1],
                object_vertex_position_after_one_mate[
                    mating_edge_index % len(current_object_vertex)
                ],
            ]
            mating_environment_edge = [
                environment[len(environment) - mating_environment_index - 1],
                environment[len(environment) - mating_environment_index],
            ]
            if (
                abs(
                    np.cross(
                        generate_vector(mating_edge[0], mating_edge[1]),
                        generate_vector(
                            mating_environment_edge[0], mating_environment_edge[1]
                        ),
                    )
                )
                / (
                    LineString(mating_edge).length
                    * LineString(mating_environment_edge).length
                )
                > 0.017
            ):
                return False
            translating_distance = abs(
                CtoAB(
                    mating_environment_edge[0],
                    mating_environment_edge[1],
                    mating_edge[0],
                )
            )
            translating_environment_index = int(
                temp_target_contact_state0[
                    temp_target_contact_state0.index("E") + 1 : :
                ]
            )
            translating_environment_edge = [
                environment[len(environment) - translating_environment_index - 1],
                environment[len(environment) - translating_environment_index],
            ]
            translating_direction = generate_vector(
                translating_environment_edge[0], translating_environment_edge[1]
            )
            translating_direction = [
                translating_direction[0]
                / LineString(translating_environment_edge).length,
                translating_direction[1]
                / LineString(translating_environment_edge).length,
            ]
            object_vertex_position_after_translation = [
                [
                    vertex[0] + translating_direction[0] * translating_distance,
                    vertex[1] + translating_direction[1] * translating_distance,
                ]
                for vertex in object_vertex_position_after_one_mate
            ]
            if sorted(
                obtain_contact_state(
                    object_vertex_position_after_translation, environment
                )
            ) == sorted(target_contact_state):
                if Polygon(object_vertex_position_after_translation).within(
                    Polygon(scope)
                ):
                    return object_vertex_position_after_translation
            object_vertex_position_after_translation = [
                [
                    vertex[0] - translating_direction[0] * translating_distance,
                    vertex[1] - translating_direction[1] * translating_distance,
                ]
                for vertex in object_vertex_position_after_one_mate
            ]
            if sorted(
                obtain_contact_state(
                    object_vertex_position_after_translation, environment
                )
            ) == sorted(target_contact_state):
                if Polygon(object_vertex_position_after_translation).within(
                    Polygon(scope)
                ):
                    return object_vertex_position_after_translation
        else:
            if (
                temp_target_contact_state0[0] == "V"
                and temp_target_contact_state1[0] == "l"
            ):
                temp = temp_target_contact_state0[:]
                temp_target_contact_state0 = temp_target_contact_state1[:]
                temp_target_contact_state1 = temp[:]
            object_vertex_position_after_one_mate = (
                finding_configuration_cos_contact_state(
                    current_object_vertex,
                    environment,
                    [temp_target_contact_state0, "single_contact"],
                    scope,
                )
            )
            if not object_vertex_position_after_one_mate:
                return False
            rotate_time = 0
            while True:
                mating_vertex_index = int(
                    temp_target_contact_state1[
                        1 : temp_target_contact_state1.index("E")
                    ]
                )
                mating_environment_index = int(
                    temp_target_contact_state1[
                        temp_target_contact_state1.index("E") + 1 : :
                    ]
                )
                mating_vertex = object_vertex_position_after_one_mate[
                    mating_vertex_index - 1
                ]
                mating_environment_edge = [
                    environment[len(environment) - mating_environment_index - 1],
                    environment[len(environment) - mating_environment_index],
                ]
                translating_environment_index = int(
                    temp_target_contact_state0[
                        temp_target_contact_state0.index("E") + 1 : :
                    ]
                )
                translating_environment_edge = [
                    environment[len(environment) - translating_environment_index - 1],
                    environment[len(environment) - translating_environment_index],
                ]
                translating_direction = generate_vector(
                    translating_environment_edge[0], translating_environment_edge[1]
                )
                translating_direction = [
                    translating_direction[0]
                    / LineString(translating_environment_edge).length,
                    translating_direction[1]
                    / LineString(translating_environment_edge).length,
                ]
                intersect_point = calculate_intersect_point_AB_CD(
                    mating_environment_edge[0],
                    mating_environment_edge[1],
                    mating_vertex,
                    [
                        mating_vertex[0] + translating_direction[0],
                        mating_vertex[1] + translating_direction[1],
                    ],
                )
                try:
                    translating_distance = LineString(
                        [intersect_point, mating_vertex]
                    ).length
                except:
                    break
                object_vertex_position_after_translation = [
                    [
                        vertex[0] + translating_direction[0] * translating_distance,
                        vertex[1] + translating_direction[1] * translating_distance,
                    ]
                    for vertex in object_vertex_position_after_one_mate
                ]
                if sorted(
                    obtain_contact_state(
                        object_vertex_position_after_translation, environment
                    )
                ) == sorted(target_contact_state):
                    if Polygon(object_vertex_position_after_translation).within(
                        Polygon(scope)
                    ):
                        return object_vertex_position_after_translation
                object_vertex_position_after_translation = [
                    [
                        vertex[0] - translating_direction[0] * translating_distance,
                        vertex[1] - translating_direction[1] * translating_distance,
                    ]
                    for vertex in object_vertex_position_after_one_mate
                ]
                if sorted(
                    obtain_contact_state(
                        object_vertex_position_after_translation, environment
                    )
                ) == sorted(target_contact_state):
                    if Polygon(object_vertex_position_after_translation).within(
                        Polygon(scope)
                    ):
                        return object_vertex_position_after_translation
                if (
                    temp_target_contact_state0[0] == "V"
                    and temp_target_contact_state1[0] == "V"
                    and rotate_time < 360.0 / 1.0
                ):
                    rotate_time += 1
                    object_vertex_position_after_one_mate = [
                        point_position_after_rotation(
                            vertex,
                            object_vertex_position_after_one_mate[
                                int(
                                    temp_target_contact_state0[
                                        1 : temp_target_contact_state0.index("E")
                                    ]
                                )
                                - 1
                            ],
                            1,
                        )
                        for vertex in object_vertex_position_after_one_mate
                    ]
                else:
                    break
    return False


# time0=time.time()
# print finding_configuration_cos_contact_state([[0.5+0.4,1], [1.4+0.4,1], [1.4+0.4,1.8], [1.3+0.4,1.9], [1.3+0.4,2.4], [0.95+0.4,2.45], [0.6+0.4,2.4], [0.6+0.4,1.9], [0.5+0.4,1.8]], [[0,0.75], [0,0], [3,0], [3,0.75]], ['l1E2', 'l9E3', 'double_contact'])
# print time.time()-time0


def calculate_parameter_tip(
    current_vertex_position,
    target_contact_state,
    target_object_pose=None,
    profile_environment=[[0, 5], [0, 0], [5, 0]],
    scope=[[-0.1, -0.1], [3.1, -0.1], [3.1, 3.1], [-0.1, 3.1]],
    rotate_step=5,
    searching_used=False,
):
    current_contact_state = obtain_contact_state(
        current_vertex_position, profile_environment
    )
    contact_list = []
    if target_object_pose != None:
        rotate_step = 1
    ini_rotate_step = rotate_step
    for vertex in current_vertex_position:
        if Point(vertex).distance(LineString(profile_environment)) < 0.004:
            contact_list.append(vertex)
    # print contact_list
    for contact in contact_list:
        for direction in ["CCW", "CW"]:
            old_vertex_position = current_vertex_position
            old_vertex_position_buffer = (
                Polygon(current_vertex_position).buffer(-0.002).boundary.coords[:]
            )
            accumulate_angle = 0
            rotate_step = ini_rotate_step
            iteration = 0
            while True:
                if direction == "CCW":
                    new_vertex_position = [
                        point_position_after_rotation(vertex, contact, rotate_step)
                        for vertex in old_vertex_position
                    ]
                    new_vertex_position_buffer = [
                        point_position_after_rotation(vertex, contact, rotate_step)
                        for vertex in old_vertex_position_buffer
                    ]
                else:
                    new_vertex_position = [
                        point_position_after_rotation(vertex, contact, -rotate_step)
                        for vertex in old_vertex_position
                    ]
                    new_vertex_position_buffer = [
                        point_position_after_rotation(vertex, contact, -rotate_step)
                        for vertex in old_vertex_position_buffer
                    ]
                if target_object_pose != None:
                    if (
                        Polygon(new_vertex_position)
                        .buffer(-0.01)
                        .intersects(LineString(profile_environment))
                    ):
                        break
                    accumulate_angle = accumulate_angle + rotate_step
                    old_vertex_position = new_vertex_position
                    i = 0
                    for k in range(len(target_object_pose)):
                        if (
                            Point(target_object_pose[k]).distance(
                                Point(new_vertex_position[k])
                            )
                            < 0.01
                        ):
                            i += 1
                    if i == len(target_object_pose):
                        return contact, direction, accumulate_angle, new_vertex_position
                else:
                    if (
                        Polygon(new_vertex_position_buffer).distance(
                            LineString(profile_environment)
                        )
                        > 0
                    ):
                        accumulate_angle = accumulate_angle + rotate_step
                        old_vertex_position = new_vertex_position
                        old_vertex_position_buffer = new_vertex_position_buffer
                        new_contact_state = obtain_contact_state(
                            new_vertex_position,
                            profile_environment,
                            new_vertex_position_buffer,
                        )
                        if set(new_contact_state) == set(target_contact_state):
                            return (
                                contact,
                                direction,
                                accumulate_angle,
                                new_vertex_position,
                            )
                        elif (
                            searching_used
                            and set(new_contact_state) != set(current_contact_state)
                            and new_contact_state != ["Invalid"]
                        ):
                            break
                    else:
                        min_bound = 0
                        max_bound = rotate_step
                        while max_bound - min_bound > 0.0001:
                            rotate_step = (
                                float(min_bound)
                                + (float(max_bound) - float(min_bound)) / 2.0
                            )
                            if direction == "CCW":
                                new_vertex_position = [
                                    point_position_after_rotation(
                                        vertex, contact, rotate_step
                                    )
                                    for vertex in old_vertex_position
                                ]
                                new_vertex_position_buffer = [
                                    point_position_after_rotation(
                                        vertex, contact, rotate_step
                                    )
                                    for vertex in old_vertex_position_buffer
                                ]
                            else:
                                new_vertex_position = [
                                    point_position_after_rotation(
                                        vertex, contact, -rotate_step
                                    )
                                    for vertex in old_vertex_position
                                ]
                                new_vertex_position_buffer = [
                                    point_position_after_rotation(
                                        vertex, contact, -rotate_step
                                    )
                                    for vertex in old_vertex_position_buffer
                                ]
                            new_contact_state = obtain_contact_state(
                                new_vertex_position,
                                profile_environment,
                                new_vertex_position_buffer,
                            )
                            if sorted(new_contact_state) == sorted(
                                target_contact_state
                            ):
                                accumulate_angle = accumulate_angle + rotate_step
                                return (
                                    contact,
                                    direction,
                                    accumulate_angle,
                                    new_vertex_position,
                                )
                            elif (
                                searching_used
                                and set(new_contact_state) != set(current_contact_state)
                                and new_contact_state != ["Invalid"]
                            ):
                                break

                            if (
                                Polygon(new_vertex_position_buffer).distance(
                                    LineString(profile_environment)
                                )
                                > 0
                            ):
                                min_bound = rotate_step
                                max_bound = max_bound
                            else:
                                min_bound = min_bound
                                max_bound = rotate_step
                        break
    return "No Solution"


def calculate_parameter_push(
    current_vertex_position,
    target_contact_state,
    target_object_pose=None,
    profile_environment=[[0, 5], [0, 0], [5, 0]],
    scope=[[-0.1, -0.1], [3.1, -0.1], [3.1, 3.1], [-0.1, 3.1]],
    translate_step=0.2,
    searching_used=False,
    accuracy=0.004,
):

    current_contact_state = obtain_contact_state(
        current_vertex_position, profile_environment
    )
    contact_list = []
    contact_env_edge_list = dict()
    if target_object_pose != None:
        translate_step = 0.01
    ini_translate_step = translate_step
    for vertex in current_vertex_position:
        if Point(vertex).distance(LineString(profile_environment)) < 0.004:
            contact_env_edge_list[tuple(vertex)] = []
            contact_list.append(vertex)
            for j in range(len(profile_environment)):
                if (
                    j != len(profile_environment) - 1
                    and Point(vertex).distance(
                        LineString([profile_environment[j], profile_environment[j + 1]])
                    )
                    < accuracy
                ):
                    contact_env_edge_list[tuple(vertex)].append(
                        [profile_environment[j], profile_environment[j + 1]]
                    )
    direction_memory = []
    for contact in contact_list:
        for direction in [
            [x[1][0] - x[0][0], x[1][1] - x[0][1]]
            for x in contact_env_edge_list[tuple(contact)]
        ] + [
            [-x[1][0] + x[0][0], -x[1][1] + x[0][1]]
            for x in contact_env_edge_list[tuple(contact)]
        ]:
            if direction in direction_memory:
                continue
            direction_memory.append(direction)
            direction = [
                direction[0] / sqrt(direction[0] ** 2 + direction[1] ** 2),
                direction[1] / sqrt(direction[0] ** 2 + direction[1] ** 2),
            ]
            old_vertex_position = current_vertex_position
            old_vertex_position_buffer = (
                Polygon(current_vertex_position).buffer(-0.002).boundary.coords[:]
            )
            accumulate_distance = 0
            translate_step = ini_translate_step
            iteration = 0
            while True:
                if searching_used:
                    iteration += 1
                    if iteration > 10:
                        break
                new_vertex_position = [
                    [
                        vertex[0] + translate_step * direction[0],
                        vertex[1] + translate_step * direction[1],
                    ]
                    for vertex in old_vertex_position
                ]
                new_vertex_position_buffer = [
                    [
                        vertex[0] + translate_step * direction[0],
                        vertex[1] + translate_step * direction[1],
                    ]
                    for vertex in old_vertex_position_buffer
                ]
                if target_object_pose != None:
                    if (
                        Polygon(new_vertex_position_buffer).distance(
                            LineString(profile_environment)
                        )
                        <= 0
                    ):
                        break
                    if (
                        Polygon(new_vertex_position).distance(
                            LineString(profile_environment)
                        )
                        > 0.05
                    ):
                        break
                    accumulate_distance = accumulate_distance + translate_step
                    old_vertex_position = new_vertex_position
                    old_vertex_position_buffer = new_vertex_position_buffer
                    i = 0
                    for k in range(len(target_object_pose)):
                        if (
                            Point(target_object_pose[k]).distance(
                                Point(new_vertex_position[k])
                            )
                            < 0.01
                        ):
                            i += 1
                    if i == len(target_object_pose):
                        return direction, accumulate_distance, new_vertex_position
                else:
                    if (
                        Polygon(new_vertex_position_buffer).distance(
                            LineString(profile_environment)
                        )
                        > 0
                    ):
                        if not Polygon(new_vertex_position).within(Polygon(scope)):
                            break
                        accumulate_distance = accumulate_distance + translate_step
                        old_vertex_position = new_vertex_position
                        old_vertex_position_buffer = new_vertex_position_buffer

                        new_contact_state = obtain_contact_state(
                            new_vertex_position,
                            profile_environment,
                            new_vertex_position_buffer,
                        )
                        if set(new_contact_state) == set(target_contact_state):
                            return direction, accumulate_distance, new_vertex_position
                        elif (
                            searching_used
                            and set(new_contact_state) != set(current_contact_state)
                            and new_contact_state != ["Invalid"]
                        ):
                            break
                    else:
                        min_bound = 0
                        max_bound = translate_step
                        while max_bound - min_bound > 0.0001:
                            translate_step = (
                                float(min_bound)
                                + (float(max_bound) - float(min_bound)) / 2.0
                            )
                            new_vertex_position = [
                                [
                                    vertex[0] + translate_step * direction[0],
                                    vertex[1] + translate_step * direction[1],
                                ]
                                for vertex in old_vertex_position
                            ]
                            new_vertex_position_buffer = [
                                [
                                    vertex[0] + translate_step * direction[0],
                                    vertex[1] + translate_step * direction[1],
                                ]
                                for vertex in old_vertex_position_buffer
                            ]
                            new_contact_state = obtain_contact_state(
                                new_vertex_position,
                                profile_environment,
                                new_vertex_position_buffer,
                            )
                            if sorted(new_contact_state) == sorted(
                                target_contact_state
                            ):
                                accumulate_distance = (
                                    accumulate_distance + translate_step
                                )
                                return (
                                    direction,
                                    accumulate_distance,
                                    new_vertex_position,
                                )
                            elif (
                                searching_used
                                and set(new_contact_state) != set(current_contact_state)
                                and new_contact_state != ["Invalid"]
                            ):
                                break
                            if (
                                Polygon(new_vertex_position_buffer).distance(
                                    LineString(profile_environment)
                                )
                                > 0
                            ):
                                min_bound = translate_step
                                max_bound = max_bound
                            else:
                                min_bound = min_bound
                                max_bound = translate_step
                        break
    return "No Solution"


def Tilting_slide_contact_edge_point(
    current_vertex_position, target_contact_state, profile_environment
):
    current_contact_state = obtain_contact_state(
        current_vertex_position, profile_environment
    )
    if "double_contact" not in current_contact_state:
        return "Invalid"
    E = dict()
    for i in range(len(profile_environment) - 1):
        E[i + 1] = [
            profile_environment[len(profile_environment) - i - 2],
            profile_environment[len(profile_environment) - i - 1],
        ]
    current_contact_state.remove("double_contact")
    for i in range(len(current_vertex_position)):
        if (
            re.match("V" + str(i + 1), current_contact_state[0]) != None
            and re.match("V" + str(i + 1), current_contact_state[1]) != None
        ):
            return "Invalid"
    contact_edge = dict()
    contact_edge_point = dict()
    for contact_state_element in current_contact_state:
        if re.match("V", contact_state_element) != None:
            contact_point = current_vertex_position[
                int(contact_state_element[1 : contact_state_element.index("E")]) - 1
            ]
            for i in range(len(profile_environment) - 1):
                if contact_state_element[
                    contact_state_element.index("E") + 1 : :
                ] == str(i + 1):
                    contact_edge["E" + str(i + 1)] = E[i + 1]
                    contact_edge_point["E" + str(i + 1)] = contact_point
        else:
            contact_point = [
                current_vertex_position[
                    int(contact_state_element[1 : contact_state_element.index("E")]) - 1
                ],
                current_vertex_position[
                    int(contact_state_element[1 : contact_state_element.index("E")])
                    % (len(current_vertex_position))
                ],
            ]
            for point in contact_point:
                if Point(point).distance(Point(profile_environment[1])) < 0.01:
                    contact_point.remove(point)
            nearest_contact_point = contact_point[0]
            for point in contact_point:
                if Point(point).distance(Point(profile_environment[1])) < Point(
                    nearest_contact_point
                ).distance(Point(profile_environment[1])):
                    nearest_contact_point = point
            contact_point = nearest_contact_point
            for i in range(len(profile_environment) - 1):
                if contact_state_element[
                    contact_state_element.index("E") + 1 : :
                ] == str(i + 1):
                    contact_edge["E" + str(i + 1)] = E[i + 1]
                    contact_edge_point["E" + str(i + 1)] = contact_point
    if "double_contact" in target_contact_state:
        target_contact_state.remove("double_contact")
    if (
        re.match("V", current_contact_state[0]) == None
        and re.match("V", current_contact_state[1]) == None
    ):
        if (
            re.match("V", target_contact_state[0]) == None
            and re.match("V", target_contact_state[1]) == None
        ):
            return "Invalid"
        else:
            for contact_state_element in target_contact_state:
                if re.match("V", contact_state_element) != None:
                    contact_point = current_vertex_position[
                        int(contact_state_element[1 : contact_state_element.index("E")])
                        - 1
                    ]
                    for i in range(len(profile_environment) - 1):
                        if contact_state_element[
                            contact_state_element.index("E") + 1 : :
                        ] == str(i + 1):
                            if (
                                Point(contact_point).distance(LineString(E[i + 1]))
                                > 0.004
                            ):
                                return "Invalid"
                            contact_edge["E" + str(i + 1)] = E[i + 1]
                            contact_edge_point["E" + str(i + 1)] = contact_point
    if "double_contact" not in target_contact_state:
        target_contact_state.append("double_contact")
    return contact_edge_point


def Tilting_slide_calculate_next_vertex_position_simplified(
    contact_edge_point, total_angle, old_vertex_position, profile_environment
):
    E = dict()
    E_direction = dict()
    for i in range(len(profile_environment) - 1):
        E[i + 1] = [
            profile_environment[len(profile_environment) - i - 2],
            profile_environment[len(profile_environment) - i - 1],
        ]
        E_direction[i + 1] = [
            (E[i + 1][1][0] - E[i + 1][0][0])
            / Point(E[i + 1][0]).distance(Point(E[i + 1][1])),
            (E[i + 1][1][1] - E[i + 1][0][1])
            / Point(E[i + 1][0]).distance(Point(E[i + 1][1])),
        ]
    E_key_set = sorted([i for i in contact_edge_point])
    Ea_direction = [
        (E[int(E_key_set[0][-1])][1][0] - E[int(E_key_set[0][-1])][0][0])
        / Point(E[int(E_key_set[0][-1])][0]).distance(
            Point(E[int(E_key_set[0][-1])][1])
        ),
        (E[int(E_key_set[0][-1])][1][1] - E[int(E_key_set[0][-1])][0][1])
        / Point(E[int(E_key_set[0][-1])][0]).distance(
            Point(E[int(E_key_set[0][-1])][1])
        ),
    ]
    Eb_direction = [
        (E[int(E_key_set[1][-1])][0][0] - E[int(E_key_set[1][-1])][1][0])
        / Point(E[int(E_key_set[1][-1])][0]).distance(
            Point(E[int(E_key_set[1][-1])][1])
        ),
        (E[int(E_key_set[1][-1])][0][1] - E[int(E_key_set[1][-1])][1][1])
        / Point(E[int(E_key_set[1][-1])][0]).distance(
            Point(E[int(E_key_set[1][-1])][1])
        ),
    ]
    intersect_angle_Ea = abs(
        calculate_intersect_angle_AB_CD(
            contact_edge_point[E_key_set[1]],
            contact_edge_point[E_key_set[0]],
            E[int(E_key_set[0][-1])][0],
            E[int(E_key_set[0][-1])][1],
        )
    )
    intersect_angle_Eb = abs(
        calculate_intersect_angle_AB_CD(
            contact_edge_point[E_key_set[0]],
            contact_edge_point[E_key_set[1]],
            E[int(E_key_set[1][-1])][1],
            E[int(E_key_set[1][-1])][0],
        )
    )
    intersect_angleEaEb = abs(
        calculate_intersect_angle_AB_CD(
            E[int(E_key_set[1][-1])][0],
            E[int(E_key_set[1][-1])][1],
            E[int(E_key_set[0][-1])][0],
            E[int(E_key_set[0][-1])][1],
        )
    )
    EaEbcontact_distance = Point(contact_edge_point[E_key_set[1]]).distance(
        Point(contact_edge_point[E_key_set[0]])
    )
    target_intersect_angle_Ea = intersect_angle_Ea - total_angle
    target_intersect_angle_Eb = intersect_angle_Eb + total_angle
    Ebdistance = (
        sin(target_intersect_angle_Ea * pi / 180)
        * EaEbcontact_distance
        / sin(intersect_angleEaEb * pi / 180)
    )
    Eadistance = (
        sin(target_intersect_angle_Eb * pi / 180)
        * EaEbcontact_distance
        / sin(intersect_angleEaEb * pi / 180)
    )
    new_Ea_contact_point = [
        E[int(E_key_set[0][-1])][0][0] + Eadistance * Ea_direction[0],
        E[int(E_key_set[0][-1])][0][1] + Eadistance * Ea_direction[1],
    ]
    new_Eb_contact_point = [
        E[int(E_key_set[0][-1])][0][0] + Ebdistance * Eb_direction[0],
        E[int(E_key_set[0][-1])][0][1] + Ebdistance * Eb_direction[1],
    ]
    delta_translate, D_r, A_r = calculate_parameter_translate_and_rotate_moveA(
        contact_edge_point[E_key_set[0]],
        contact_edge_point[E_key_set[1]],
        new_Ea_contact_point,
        new_Eb_contact_point,
    )
    A_r = A_r * 180 / pi
    new_vertex_position = [
        [
            point_position_after_rotation(
                vertex, contact_edge_point[E_key_set[0]], A_r
            )[0]
            + delta_translate[0],
            point_position_after_rotation(
                vertex, contact_edge_point[E_key_set[0]], A_r
            )[1]
            + delta_translate[1],
        ]
        for vertex in old_vertex_position
    ]
    return new_vertex_position


def calculate_parameter_tilting_slide(
    current_vertex_position,
    target_contact_state,
    target_object_pose=None,
    profile_environment=[[0, 5], [0, 0], [5, 0]],
    scope=[[-0.1, -0.1], [3.1, -0.1], [3.1, 3.1], [-0.1, 3.1]],
    rotate_step=30,
    searching_used=False,
):
    current_contact_state = obtain_contact_state(
        current_vertex_position, profile_environment
    )
    print(current_contact_state)
    if target_object_pose != None:
        rotate_step = 1
    # try:
    # print Tilting_slide_contact_edge_point(current_vertex_position, target_contact_state, profile_environment)
    contact_edge_point = Tilting_slide_contact_edge_point(
        current_vertex_position, target_contact_state, profile_environment
    )
    # except:
    # print 'ccc'
    # return 'No Solution'

    for direction in ["+", "-"]:
        A_r = 0
        rotate_step0 = rotate_step
        while True:
            if direction == "+":
                rotate_step_real = abs(rotate_step)
            else:
                rotate_step_real = -abs(rotate_step)
            A_r = A_r + rotate_step_real
            try:
                new_vertex_position = (
                    Tilting_slide_calculate_next_vertex_position_simplified(
                        contact_edge_point,
                        A_r,
                        current_vertex_position,
                        profile_environment,
                    )
                )
            except:
                return "No Solution"
            if target_object_pose != None:
                if (
                    Polygon(new_vertex_position)
                    .buffer(-0.002)
                    .crosses(LineString(profile_environment))
                ):
                    break
                i = 0
                for k in range(len(target_object_pose)):
                    if (
                        Point(target_object_pose[k]).distance(
                            Point(new_vertex_position[k])
                        )
                        < 0.01
                    ):
                        i += 1
                if i == len(target_object_pose):
                    return A_r, new_vertex_position
            else:
                if (
                    not Polygon(new_vertex_position)
                    .buffer(-0.002)
                    .crosses(LineString(profile_environment))
                ):
                    new_contact_state = obtain_contact_state(
                        new_vertex_position, profile_environment
                    )
                    if set(new_contact_state) == set(target_contact_state):
                        return A_r, new_vertex_position
                    elif (
                        searching_used
                        and set(new_contact_state) != set(current_contact_state)
                        and new_contact_state != ["Invalid"]
                    ):
                        break
                else:
                    min_bound = 0
                    max_bound = rotate_step0
                    while max_bound - min_bound > 0.0001:
                        rotate_step0 = (
                            float(min_bound)
                            + (float(max_bound) - float(min_bound)) / 2.0
                        )
                        if direction == "+":
                            rotate_angle = A_r - rotate_step_real + rotate_step0
                            try:
                                new_vertex_position = Tilting_slide_calculate_next_vertex_position_simplified(
                                    contact_edge_point,
                                    rotate_angle,
                                    current_vertex_position,
                                    profile_environment,
                                )
                            except:
                                return "No Solution"
                        else:
                            rotate_angle = A_r - rotate_step_real - rotate_step0
                            try:
                                new_vertex_position = Tilting_slide_calculate_next_vertex_position_simplified(
                                    contact_edge_point,
                                    rotate_angle,
                                    current_vertex_position,
                                    profile_environment,
                                )
                            except:
                                return "No Solution"

                        new_contact_state = obtain_contact_state(
                            new_vertex_position, profile_environment
                        )
                        if sorted(new_contact_state) == sorted(target_contact_state):
                            return rotate_angle, new_vertex_position
                        elif (
                            searching_used
                            and set(new_contact_state) != set(current_contact_state)
                            and new_contact_state != ["Invalid"]
                        ):
                            temp_new_contact_state = new_contact_state[:]
                            if "double_contact" not in temp_new_contact_state:
                                return "No Solution"
                            temp_new_contact_state.remove("double_contact")
                            if (
                                temp_new_contact_state[0][0] == "V"
                                and temp_new_contact_state[1][0] == "V"
                            ):
                                break

                        if (
                            not Polygon(new_vertex_position)
                            .buffer(-0.002)
                            .crosses(LineString(profile_environment))
                        ):
                            min_bound = rotate_step0
                            max_bound = max_bound
                        else:
                            min_bound = min_bound
                            max_bound = rotate_step0
                    break
    return "No Solution"


# print calculate_parameter_tilting_slide([[3,1.5], [2.3,1.5], [2.3,0], [3,0]], ['double_contact', 'V4E2', 'V1E1'], profile_environment = [[0,0.75], [0,0], [3,0], [3,0.75]])

# print calculate_parameter_tilting_slide([[-1.1102230246251565e-16, 0.4499999999999999], [0.7794228634059949, 0.0], [1.1794228634059947, 0.692820323027551], [1.142820323027551, 0.8294228634059947], [1.392820323027551, 1.262435565298214], [1.1147114317029974, 1.480736835487436], [0.7866025403784437, 1.612435565298214], [0.5366025403784438, 1.1794228634059947], [0.3999999999999998, 1.1428203230275509]], ['double_contact', 'l1E3', 'l2E2'], None, [[0, 0.75], [0, 0], [3, 0], [3, 0.75]])


def calculate_parameter_move_in_air(
    current_vertex_position,
    target_contact_state,
    target_object_pose=None,
    moving_direction="up",
    profile_environment=[[0, 5], [0, 0], [5, 0]],
    translate_step=0.2,
):
    if target_object_pose != None:
        translate_step = 0.01
    ini_translate_step = translate_step
    if moving_direction == "up":
        direction = [0, 1]
    elif moving_direction == "down":
        direction = [0, -1]
    elif moving_direction == "left":
        direction = [-1, 0]
    elif moving_direction == "right":
        direction = [1, 0]
    old_vertex_position = current_vertex_position
    accumulate_distance = 0
    translate_step = ini_translate_step
    while True:
        new_vertex_position = [
            [
                vertex[0] + translate_step * direction[0],
                vertex[1] + translate_step * direction[1],
            ]
            for vertex in old_vertex_position
        ]
        if target_object_pose != None:
            if (
                Polygon(new_vertex_position)
                .buffer(-0.002)
                .intersects(LineString(profile_environment))
            ):
                break
            accumulate_distance = accumulate_distance + translate_step
            old_vertex_position = new_vertex_position
            i = 0
            for k in range(len(target_object_pose)):
                if (
                    Point(target_object_pose[k]).distance(Point(new_vertex_position[k]))
                    < 0.01
                ):
                    i += 1
                if i == len(target_object_pose):
                    return direction, accumulate_distance, new_vertex_position
        else:
            if (
                not Polygon(new_vertex_position)
                .buffer(-0.002)
                .crosses(LineString(profile_environment))
            ):
                accumulate_distance = accumulate_distance + translate_step
                old_vertex_position = new_vertex_position
                if set(
                    obtain_contact_state(new_vertex_position, profile_environment)
                ) == set(target_contact_state):
                    return direction, accumulate_distance, new_vertex_position
                else:
                    min_bound = 0
                    max_bound = translate_step
                    while max_bound - min_bound > 0.0001:
                        translate_step = (
                            float(min_bound)
                            + (float(max_bound) - float(min_bound)) / 2.0
                        )
                        new_vertex_position = [
                            [
                                vertex[0] + translate_step * direction[0],
                                vertex[1] + translate_step * direction[1],
                            ]
                            for vertex in old_vertex_position
                        ]
                        if set(
                            obtain_contact_state(
                                new_vertex_position, profile_environment
                            )
                        ) == set(target_contact_state):
                            accumulate_distance = accumulate_distance + translate_step
                            return direction, accumulate_distance, new_vertex_position
                        if (
                            not Polygon(new_vertex_position)
                            .buffer(-0.002)
                            .crosses(LineString(profile_environment))
                        ):
                            min_bound = translate_step
                            max_bound = max_bound
                        else:
                            min_bound = min_bound
                            max_bound = translate_step
                    break
    return "No Solution"


def Tip_next_vertex_position(
    current_object_vertex_position, action_parameter, profile_environment
):
    [rotation_pole, rotation_direction, rotation_angle] = action_parameter
    if rotation_direction == "CCW":
        new_vertex_position = [
            point_position_after_rotation(vertex, rotation_pole, rotation_angle)
            for vertex in current_object_vertex_position
        ]
    else:
        new_vertex_position = [
            point_position_after_rotation(vertex, rotation_pole, -rotation_angle)
            for vertex in current_object_vertex_position
        ]
    return new_vertex_position


def Push_next_vertex_position(
    current_object_vertex_position, action_parameter, profile_environment
):
    [push_direction, push_distance] = action_parameter
    new_vertex_position = [
        [
            vertex[0] + push_distance * push_direction[0],
            vertex[1] + push_distance * push_direction[1],
        ]
        for vertex in current_object_vertex_position
    ]
    return new_vertex_position


def Move_in_air_next_vertex_position(
    current_object_vertex_position, action_parameter, profile_environment
):
    [move_direction, move_distance] = action_parameter
    new_vertex_position = [
        [
            vertex[0] + move_distance * move_direction[0],
            vertex[1] + move_distance * move_direction[1],
        ]
        for vertex in current_object_vertex_position
    ]
    return new_vertex_position


def Tilting_slide_next_vertex_position(
    current_object_vertex_position, action_parameter, profile_environment
):
    [target_contact_state, A_r] = action_parameter
    contact_edge_point = Tilting_slide_contact_edge_point(
        current_object_vertex_position, target_contact_state, profile_environment
    )
    new_vertex_position = Tilting_slide_calculate_next_vertex_position_simplified(
        contact_edge_point, A_r, current_object_vertex_position, profile_environment
    )
    return new_vertex_position


class contact_state:
    def __init__(self, environment, initial_vertex_position):
        (
            self.contact_state_list,
            self.next_contact_state,
            self.action_to_next_contact_state,
            self.action_cost,
        ) = self.contact_state_map(environment, initial_vertex_position)
        # print self.contact_state_list, self.next_contact_state, self.action_to_next_contact_state, self.action_cost
        # print len(self.contact_state_list)
        # print self.contact_state_list, self.next_contact_state, self.action_to_next_contact_state, self.action_cost

        # n = open('task4contactstatemap.txt','w')
        # n.writelines('\n'+'ggg'+str(self.contact_state_list))
        # n.writelines('\n'+'hhh'+str(self.next_contact_state))
        # n.writelines('\n'+'iii'+str(self.action_to_next_contact_state))
        # n.writelines('\n'+'jjj'+str(self.action_cost))
        # +'[dir=\"both\"]'

        """
        n = open('task3contactstatemap.dot','w')
        n.writelines("digraph A {")
        n.writelines('\n'+'rankdir = LR')
        transition_memory=[]
        for contact_state in self.contact_state_list:
            current_next_contact_state = self.next_contact_state[tuple(contact_state)]
            current_action_to_next_state = self.action_to_next_contact_state[tuple(contact_state)]
            if 'double_contact' in contact_state:
                contact_state.remove('double_contact')
                prefix = '\n'+'\t'+'\"'+contact_state[0]+','+contact_state[1]+'\"'
                n.writelines(prefix)
            if 'single_contact' in contact_state:
                contact_state.remove('single_contact')
                prefix = '\n'+'\t'+'\"'+contact_state[0]+'\"'
                n.writelines(prefix)
            for j in range(len(current_next_contact_state)):
                temp_next_contact_state = list(current_next_contact_state[j])
                if 'double_contact' in temp_next_contact_state:
                    temp_next_contact_state.remove('double_contact')
                    if [temp_next_contact_state, contact_state, current_action_to_next_state[j]] in transition_memory:
                        continue
                    transition_memory.append([contact_state, temp_next_contact_state, current_action_to_next_state[j]])
                    if current_action_to_next_state[j]=='Tip':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+','+temp_next_contact_state[1]+'\"'+'[color=forestgreen]'+'[arrowsize=0.6]'+'[dir=both]')
                    elif current_action_to_next_state[j]=='Push':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+','+temp_next_contact_state[1]+'\"'+'[color=darkorange1]'+'[arrowsize=0.6]'+'[dir=both]')
                    elif current_action_to_next_state[j]=='Tilting-slide':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+','+temp_next_contact_state[1]+'\"'+'[color=dodgerblue]'+'[arrowsize=0.6]'+'[dir=both]')
                if 'single_contact' in temp_next_contact_state:
                    temp_next_contact_state.remove('single_contact')
                    if [temp_next_contact_state, contact_state, current_action_to_next_state[j]] in transition_memory:
                        continue
                    transition_memory.append([contact_state, temp_next_contact_state, current_action_to_next_state[j]])
                    if current_action_to_next_state[j]=='Tip':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+'\"'+'[color=forestgreen]'+'[arrowsize=0.6]'+'[dir=both]')
                    if current_action_to_next_state[j]=='Push':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+'\"'+'[color=darkorange1]'+'[arrowsize=0.6]'+'[dir=both]')
                    if current_action_to_next_state[j]=='Tilting-slide':
                        n.writelines(prefix+'->'+'\"'+temp_next_contact_state[0]+'\"'+'[color=dodgerblue]'+'[arrowsize=0.6]'+'[dir=both]')
                
        n.writelines('\n'+"}")
        """

    def contact_state_map(
        self,
        environment,
        initial_vertex_position,
        scope=[[-0.1, -0.1], [3.1, -0.1], [3.1, 3.1], [-0.1, 3.1]],
    ):
        time0 = time.time()
        state_position_pair_state_list = []
        state_position_pair_position_list = []
        object_edge_number = len(initial_vertex_position)
        environment_edge_number = len(environment) - 1
        potential_contact_state_set = []
        for edge_index in range(1, object_edge_number + 1):
            for environment_edge_index in range(1, environment_edge_number + 1):
                if (
                    finding_configuration_cos_contact_state(
                        initial_vertex_position,
                        environment,
                        [
                            "l" + str(edge_index) + "E" + str(environment_edge_index),
                            "single_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    potential_contact_state_set.append(
                        sorted(
                            [
                                "l"
                                + str(edge_index)
                                + "E"
                                + str(environment_edge_index),
                                "single_contact",
                            ]
                        )
                    )
                if (
                    finding_configuration_cos_contact_state(
                        initial_vertex_position,
                        environment,
                        [
                            "V" + str(edge_index) + "E" + str(environment_edge_index),
                            "single_contact",
                        ],
                        scope,
                    )
                    != False
                ):
                    potential_contact_state_set.append(
                        sorted(
                            [
                                "V"
                                + str(edge_index)
                                + "E"
                                + str(environment_edge_index),
                                "single_contact",
                            ]
                        )
                    )
                for edge_index2 in range(1, object_edge_number + 1):
                    for environment_edge_index2 in range(
                        1, environment_edge_number + 1
                    ):
                        # print initial_vertex_position, environment, ['l'+str(edge_index)+'E'+str(environment_edge_index), 'l'+str(edge_index2)+'E'+str(environment_edge_index2), 'double_contact']
                        if (
                            sorted(
                                [
                                    "l"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "l"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ]
                            )
                            not in potential_contact_state_set
                            and finding_configuration_cos_contact_state(
                                initial_vertex_position,
                                environment,
                                [
                                    "l"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "l"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ],
                                scope,
                            )
                            != False
                        ):
                            potential_contact_state_set.append(
                                sorted(
                                    [
                                        "l"
                                        + str(edge_index)
                                        + "E"
                                        + str(environment_edge_index),
                                        "l"
                                        + str(edge_index2)
                                        + "E"
                                        + str(environment_edge_index2),
                                        "double_contact",
                                    ]
                                )
                            )
                        # print 'bbb', ['l'+str(edge_index)+'E'+str(environment_edge_index), 'V'+str(edge_index2)+'E'+str(environment_edge_index2), 'double_contact']
                        if (
                            sorted(
                                [
                                    "l"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "V"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ]
                            )
                            not in potential_contact_state_set
                            and finding_configuration_cos_contact_state(
                                initial_vertex_position,
                                environment,
                                [
                                    "l"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "V"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ],
                                scope,
                            )
                            != False
                        ):
                            potential_contact_state_set.append(
                                sorted(
                                    [
                                        "l"
                                        + str(edge_index)
                                        + "E"
                                        + str(environment_edge_index),
                                        "V"
                                        + str(edge_index2)
                                        + "E"
                                        + str(environment_edge_index2),
                                        "double_contact",
                                    ]
                                )
                            )
                        # print initial_vertex_position, environment, ['V'+str(edge_index)+'E'+str(environment_edge_index), 'V'+str(edge_index2)+'E'+str(environment_edge_index2), 'double_contact']
                        if (
                            sorted(
                                [
                                    "V"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "V"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ]
                            )
                            not in potential_contact_state_set
                            and finding_configuration_cos_contact_state(
                                initial_vertex_position,
                                environment,
                                [
                                    "V"
                                    + str(edge_index)
                                    + "E"
                                    + str(environment_edge_index),
                                    "V"
                                    + str(edge_index2)
                                    + "E"
                                    + str(environment_edge_index2),
                                    "double_contact",
                                ],
                                scope,
                            )
                            != False
                        ):
                            potential_contact_state_set.append(
                                sorted(
                                    [
                                        "V"
                                        + str(edge_index)
                                        + "E"
                                        + str(environment_edge_index),
                                        "V"
                                        + str(edge_index2)
                                        + "E"
                                        + str(environment_edge_index2),
                                        "double_contact",
                                    ]
                                )
                            )
        potential_contact_state_set_double_first = []
        potential_contact_state_set_single_last = []
        for k in range(len(potential_contact_state_set)):
            if "double_contact" in potential_contact_state_set[k]:
                potential_contact_state_set_double_first.append(
                    potential_contact_state_set[k]
                )
            else:
                potential_contact_state_set_single_last.append(
                    potential_contact_state_set[k]
                )
        potential_contact_state_set = (
            potential_contact_state_set_double_first
            + potential_contact_state_set_single_last
        )
        next_contact_state = dict()
        action_to_next_contact_state = dict()
        action_cost = dict()
        for i in range(len(potential_contact_state_set)):
            for j in range(len(potential_contact_state_set)):
                if i == j:
                    continue
                if not whether_less_constraint(
                    potential_contact_state_set[i],
                    potential_contact_state_set[j],
                    len(initial_vertex_position),
                ) and not whether_less_constraint(
                    potential_contact_state_set[j],
                    potential_contact_state_set[i],
                    len(initial_vertex_position),
                ):
                    continue
                if (
                    "double_contact" in potential_contact_state_set[i]
                    and "double_contact" in potential_contact_state_set[j]
                ):
                    temp_potential_contact_state_set_i = potential_contact_state_set[i][
                        :
                    ]
                    temp_potential_contact_state_set_i.remove("double_contact")
                    temp_potential_contact_state_set_j = potential_contact_state_set[j][
                        :
                    ]
                    temp_potential_contact_state_set_j.remove("double_contact")
                    if (
                        temp_potential_contact_state_set_i[0][
                            0 : temp_potential_contact_state_set_i[0].index("E")
                        ]
                        == temp_potential_contact_state_set_i[1][
                            0 : temp_potential_contact_state_set_i[1].index("E")
                        ]
                        or temp_potential_contact_state_set_j[0][
                            0 : temp_potential_contact_state_set_j[0].index("E")
                        ]
                        == temp_potential_contact_state_set_j[1][
                            0 : temp_potential_contact_state_set_j[1].index("E")
                        ]
                    ):
                        if tuple(potential_contact_state_set[i]) in next_contact_state:
                            already_have = False
                            for k0 in range(
                                len(
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ]
                                )
                            ):
                                if (
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == tuple(potential_contact_state_set[j])
                                    and action_to_next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == "Tip"
                                ):
                                    already_have = True
                                    break
                            if already_have:
                                continue
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append(tuple(potential_contact_state_set[j]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append("Tip")
                            action_cost[tuple(potential_contact_state_set[i])].append(1)
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Tip")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Tip"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = [tuple(potential_contact_state_set[j])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = ["Tip"]
                            action_cost[tuple(potential_contact_state_set[i])] = [1]
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Tip")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Tip"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                    else:
                        if tuple(potential_contact_state_set[i]) in next_contact_state:
                            already_have = False
                            for k0 in range(
                                len(
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ]
                                )
                            ):
                                if (
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == tuple(potential_contact_state_set[j])
                                    and action_to_next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == "Tilting-slide"
                                ):
                                    already_have = True
                                    break
                            if already_have:
                                continue
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append(tuple(potential_contact_state_set[j]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append("Tilting-slide")
                            action_cost[tuple(potential_contact_state_set[i])].append(1)
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Tilting-slide")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Tilting-slide"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = [tuple(potential_contact_state_set[j])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = ["Tilting-slide"]
                            action_cost[tuple(potential_contact_state_set[i])] = [1]
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Tilting-slide")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Tilting-slide"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                elif (
                    "single_contact" in potential_contact_state_set[i]
                    and "single_contact" in potential_contact_state_set[j]
                ):
                    if tuple(potential_contact_state_set[i]) in next_contact_state:
                        already_have = False
                        for k0 in range(
                            len(
                                next_contact_state[
                                    tuple(potential_contact_state_set[i])
                                ]
                            )
                        ):
                            if (
                                next_contact_state[
                                    tuple(potential_contact_state_set[i])
                                ][k0]
                                == tuple(potential_contact_state_set[j])
                                and action_to_next_contact_state[
                                    tuple(potential_contact_state_set[i])
                                ][k0]
                                == "Tip"
                            ):
                                already_have = True
                                break
                        if already_have:
                            continue
                        next_contact_state[
                            tuple(potential_contact_state_set[i])
                        ].append(tuple(potential_contact_state_set[j]))
                        action_to_next_contact_state[
                            tuple(potential_contact_state_set[i])
                        ].append("Tip")
                        action_cost[tuple(potential_contact_state_set[i])].append(1)
                        if tuple(potential_contact_state_set[j]) in next_contact_state:
                            next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ].append(tuple(potential_contact_state_set[i]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ].append("Tip")
                            action_cost[tuple(potential_contact_state_set[j])].append(1)
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ] = [tuple(potential_contact_state_set[i])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ] = ["Tip"]
                            action_cost[tuple(potential_contact_state_set[j])] = [1]
                    else:
                        next_contact_state[tuple(potential_contact_state_set[i])] = [
                            tuple(potential_contact_state_set[j])
                        ]
                        action_to_next_contact_state[
                            tuple(potential_contact_state_set[i])
                        ] = ["Tip"]
                        action_cost[tuple(potential_contact_state_set[i])] = [1]
                        if tuple(potential_contact_state_set[j]) in next_contact_state:
                            next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ].append(tuple(potential_contact_state_set[i]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ].append("Tip")
                            action_cost[tuple(potential_contact_state_set[j])].append(1)
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ] = [tuple(potential_contact_state_set[i])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[j])
                            ] = ["Tip"]
                            action_cost[tuple(potential_contact_state_set[j])] = [1]
                else:
                    two_contact_State_intersection = list(
                        set(potential_contact_state_set[i]).intersection(
                            set(potential_contact_state_set[j])
                        )
                    )
                    if two_contact_State_intersection != []:
                        if tuple(potential_contact_state_set[i]) in next_contact_state:
                            already_have = False
                            for k0 in range(
                                len(
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ]
                                )
                            ):
                                if (
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == tuple(potential_contact_state_set[j])
                                    and action_to_next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == "Push"
                                ):
                                    already_have = True
                                    break
                            if already_have:
                                continue
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append(tuple(potential_contact_state_set[j]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append("Push")
                            action_cost[tuple(potential_contact_state_set[i])].append(1)
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Push")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Push"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = [tuple(potential_contact_state_set[j])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = ["Push"]
                            action_cost[tuple(potential_contact_state_set[i])] = [1]
                            if (
                                tuple(potential_contact_state_set[j])
                                in next_contact_state
                            ):
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append(tuple(potential_contact_state_set[i]))
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ].append("Push")
                                action_cost[
                                    tuple(potential_contact_state_set[j])
                                ].append(1)
                            else:
                                next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = [tuple(potential_contact_state_set[i])]
                                action_to_next_contact_state[
                                    tuple(potential_contact_state_set[j])
                                ] = ["Push"]
                                action_cost[tuple(potential_contact_state_set[j])] = [1]
                    if "double_contact" in potential_contact_state_set[i]:
                        double_contact_contact_state = potential_contact_state_set[i][:]
                        double_contact_contact_state.remove("double_contact")
                    if "double_contact" in potential_contact_state_set[j]:
                        double_contact_contact_state = potential_contact_state_set[j][:]
                        double_contact_contact_state.remove("double_contact")
                    if (
                        (
                            two_contact_State_intersection == []
                            or (
                                "V" in two_contact_State_intersection[0]
                                and double_contact_contact_state[0][
                                    0 : double_contact_contact_state[0].index("E")
                                ]
                                != double_contact_contact_state[1][
                                    0 : double_contact_contact_state[1].index("E")
                                ]
                            )
                        )
                        and "double_contact" in potential_contact_state_set[i]
                        and "single_contact" in potential_contact_state_set[j]
                    ):
                        if tuple(potential_contact_state_set[i]) in next_contact_state:
                            already_have = False
                            for k0 in range(
                                len(
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ]
                                )
                            ):
                                if (
                                    next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == tuple(potential_contact_state_set[j])
                                    and action_to_next_contact_state[
                                        tuple(potential_contact_state_set[i])
                                    ][k0]
                                    == "Tip"
                                ):
                                    already_have = True
                                    break
                            if already_have:
                                continue
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append(tuple(potential_contact_state_set[j]))
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ].append("Tip")
                            action_cost[tuple(potential_contact_state_set[i])].append(1)
                        else:
                            next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = [tuple(potential_contact_state_set[j])]
                            action_to_next_contact_state[
                                tuple(potential_contact_state_set[i])
                            ] = ["Tip"]
                            action_cost[tuple(potential_contact_state_set[i])] = [1]

        return (
            potential_contact_state_set,
            next_contact_state,
            action_to_next_contact_state,
            action_cost,
        )

    def contact_state_plan(self, ini_contact_state, final_contact_state):
        contact_state_list_temp = self.contact_state_list[:]
        contact_state_list = []
        for contact_state in contact_state_list_temp:
            contact_state_list.append(tuple(sorted(contact_state)))
        OPEN_set = {tuple(sorted(ini_contact_state))}
        CLOSED = set()
        past_cost = dict.fromkeys(contact_state_list, float("inf"))
        past_cost[tuple(sorted(ini_contact_state))] = 0
        parent = dict()
        previous_action = dict()
        while OPEN_set != set():
            min_past_cost = past_cost[list(OPEN_set)[0]]
            first_node = list(OPEN_set)[0]
            for node in list(OPEN_set):
                if past_cost[node] < min_past_cost:
                    min_past_cost = past_cost[node]
                    first_node = node
            OPEN_set.remove(first_node)
            CLOSED.add(first_node)
            if set(first_node) == set(final_contact_state):
                final_node = first_node
                break
            for contact_state, action_list in self.action_to_next_contact_state.items():
                if tuple(sorted(contact_state)) == first_node:
                    first_node_action_set = action_list
                    break
            for (
                contact_state,
                next_contact_state_list,
            ) in self.next_contact_state.items():
                if tuple(sorted(contact_state)) == first_node:
                    first_node_next_contact_list = next_contact_state_list
                    break
            for contact_state, action_cost_list in self.action_cost.items():
                if tuple(sorted(contact_state)) == first_node:
                    first_node_action_cost = action_cost_list[:]
                    break
            for i in range(len(first_node_action_set)):
                if tuple(sorted(first_node_next_contact_list[i])) in CLOSED:
                    continue
                tentative_past_cost = past_cost[first_node] + first_node_action_cost[i]
                if (
                    tentative_past_cost
                    < past_cost[tuple(sorted(first_node_next_contact_list[i]))]
                ):
                    past_cost[tuple(sorted(first_node_next_contact_list[i]))] = (
                        tentative_past_cost
                    )
                    parent[tuple(sorted(first_node_next_contact_list[i]))] = first_node
                    previous_action[tuple(sorted(first_node_next_contact_list[i]))] = (
                        first_node_action_set[i]
                    )
                    OPEN_set.add(tuple(sorted(first_node_next_contact_list[i])))
        previous_action_total = []
        contact_state_trajectory = []
        while final_node != tuple(sorted(ini_contact_state)):
            previous_action_total.append(previous_action[final_node])
            contact_state_trajectory.append(final_node)
            final_node = parent[final_node]
        previous_action_total = previous_action_total[::-1]
        contact_state_trajectory = xlist(contact_state_trajectory[::-1])
        return previous_action_total, contact_state_trajectory


# a=contact_state([[0,0.75], [0,0], [3,0], [3,0.75]], [[0.5,0], [1.2,0], [1.2,1.5], [0.5,1.5]])
# time0=time.time()
# print a.contact_state_map([[0,0.75], [0,0], [3,0], [3,0.75]], [[0.5,0], [1.2,0], [1.2,1.5], [0.5,1.5]])
# print time.time()-time0


def object_pose_plan(
    initial_object_vertex_position,
    final_object_vertex_position,
    action_list,
    contact_state_trajectory,
    profile_environment,
    repeating_time=0,
):
    begin_object_vertex_position = initial_object_vertex_position
    action_parameter = []
    object_pose_trajectory = []
    if repeating_time != 0 and len(action_list) <= 1:
        return "Infeasible"
    if repeating_time != 0 and len(action_list) > 1:
        random.seed(5)
        adjust_action_index = random.randint(0, len(action_list) - 1)
    for i in range(len(action_list)):
        if action_list[i] == "Tip":
            try:
                if i != len(action_list) - 1:
                    (
                        rotation_pole,
                        rotation_direction,
                        rotation_angle,
                        new_vertex_position,
                    ) = calculate_parameter_tip(
                        begin_object_vertex_position,
                        list(contact_state_trajectory[i]),
                        None,
                        profile_environment,
                    )
                    if repeating_time != 0 and i == adjust_action_index:
                        rotation_angle = rotation_angle + 10 * random.randint(1, 4)
                        if rotation_direction == "CCW":
                            new_vertex_position = [
                                point_position_after_rotation(
                                    vertex, rotation_pole, rotation_angle
                                )
                                for vertex in bgin_object_vertex_position
                            ]
                        else:
                            new_vertex_position = [
                                point_position_after_rotation(
                                    vertex, rotation_pole, -rotation_angle
                                )
                                for vertex in begin_object_vertex_position
                            ]
                        if obtain_contact_state(
                            new_vertex_position, profile_environment
                        ) != list(contact_state_trajectory[i]):
                            # print 'aaa'
                            return "Infeasible"
                elif final_object_vertex_position != None:
                    (
                        rotation_pole,
                        rotation_direction,
                        rotation_angle,
                        new_vertex_position,
                    ) = calculate_parameter_tip(
                        begin_object_vertex_position,
                        list(contact_state_trajectory[i]),
                        final_object_vertex_position,
                        profile_environment,
                    )
                else:
                    # print begin_object_vertex_position, list(contact_state_trajectory[i]), None, profile_environment
                    (
                        rotation_pole,
                        rotation_direction,
                        rotation_angle,
                        new_vertex_position,
                    ) = calculate_parameter_tip(
                        begin_object_vertex_position,
                        list(contact_state_trajectory[i]),
                        None,
                        profile_environment,
                    )
                action_parameter.append(
                    [rotation_pole, rotation_direction, rotation_angle]
                )
                object_pose_trajectory.append(new_vertex_position)
                begin_object_vertex_position = new_vertex_position
            except:
                # print 'aaa2'
                return "Infeasible"
        elif action_list[i] == "Push":
            try:
                if i != len(action_list) - 1:
                    # print begin_object_vertex_position, list(contact_state_trajectory[i]), None, profile_environment
                    push_direction, push_distance, new_vertex_position = (
                        calculate_parameter_push(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            None,
                            profile_environment,
                        )
                    )
                    if repeating_time != 0 and i == adjust_action_index:
                        push_distance = push_distance + 0.2 * random.randint(1, 4)
                        new_vertex_position = [
                            [
                                vertex[0] + push_distance * push_direction[0],
                                vertex[1] + push_distance * push_direction[1],
                            ]
                            for vertex in begin_object_vertex_position
                        ]
                        if obtain_contact_state(
                            new_vertex_position, profile_environment
                        ) != list(contact_state_trajectory[i]):
                            # print 'bbb'
                            return "Infeasible"
                elif final_object_vertex_position != None:
                    push_direction, push_distance, new_vertex_position = (
                        calculate_parameter_push(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            final_object_vertex_position,
                            profile_environment,
                        )
                    )
                else:
                    push_direction, push_distance, new_vertex_position = (
                        calculate_parameter_push(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            None,
                            profile_environment,
                        )
                    )
                action_parameter.append([push_direction, push_distance])
                object_pose_trajectory.append(new_vertex_position)
                begin_object_vertex_position = new_vertex_position
            except:
                # print 'eee'
                return "Infeasible"
        elif action_list[i] == "Tilting-slide":
            try:
                if i != len(action_list) - 1:
                    # print begin_object_vertex_position, list(contact_state_trajectory[i]), None, profile_environment
                    delta_angle, new_vertex_position = (
                        calculate_parameter_tilting_slide(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            None,
                            profile_environment,
                        )
                    )
                    if repeating_time != 0 and i == adjust_action_index:
                        if delta_angle > 0:
                            random.seed(5)
                            delta_angle = delta_angle + 10 * random.randint(1, 4)
                        else:
                            random.seed(5)
                            delta_angle = delta_angle - 10 * random.randint(1, 4)
                        contact_edge_point = Tilting_slide_contact_edge_point(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            profile_environment,
                        )
                        new_vertex_position = (
                            Tilting_slide_calculate_next_vertex_position_simplified(
                                contact_edge_point,
                                delta_angle,
                                begin_object_vertex_position,
                                profile_environment,
                            )
                        )
                        if obtain_contact_state(
                            new_vertex_position, profile_environment
                        ) != list(contact_state_trajectory[i]):
                            return "Infeasible"
                elif final_object_vertex_position != None:
                    delta_angle, new_vertex_position = (
                        calculate_parameter_tilting_slide(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            final_object_vertex_position,
                            profile_environment,
                        )
                    )
                else:
                    delta_angle, new_vertex_position = (
                        calculate_parameter_tilting_slide(
                            begin_object_vertex_position,
                            list(contact_state_trajectory[i]),
                            None,
                            profile_environment,
                        )
                    )
                action_parameter.append(
                    [list(contact_state_trajectory[i]), delta_angle]
                )
                object_pose_trajectory.append(new_vertex_position)
                begin_object_vertex_position = new_vertex_position
            except:
                return "Infeasible"
    return action_parameter, object_pose_trajectory


# print object_pose_plan([[0.5+0.4,0], [1.4+0.4,0], [1.4+0.4,0.8], [1.3+0.4,0.9], [1.3+0.4,1.4], [0.95+0.4,1.45], [0.6+0.4,1.4], [0.6+0.4,0.9], [0.5+0.4,0.8]], None, ['Push', 'Tilting-slide', 'Tilting-slide', 'Push'], [['double_contact', 'l1E2', 'l9E3'], ['V1E3', 'V2E2', 'double_contact'], ['double_contact', 'l1E3', 'l2E2'], ['l2E2', 'single_contact']], [[0,0.75], [0,0], [3,0], [3,0.75]], repeating_time=0)


def object_pose_plan_within_contact_state(
    initial_object_vertex_position,
    final_object_vertex_position,
    profile_environment,
    accuracy=0.004,
):
    initial_contact_state = obtain_contact_state(
        initial_object_vertex_position, profile_environment
    )
    final_contact_state = obtain_contact_state(
        final_object_vertex_position, profile_environment
    )
    if initial_contact_state != final_contact_state:
        return False
    for vertex in final_object_vertex_position:
        if Point(vertex).intersects(LineString(profile_environment).buffer(0.01)):
            final_pole = vertex
            break
    for vertex in initial_object_vertex_position:
        if Point(vertex).intersects(LineString(profile_environment).buffer(0.01)):
            pole = vertex
            break
    previous_action_total = []
    contact_state_trajectory = []
    action_parameter_trajectory = []
    object_pose_trajectory = []
    if "single_contact" in initial_contact_state:
        delta_angle = calculate_intersect_angle_AB_CD(
            initial_object_vertex_position[0],
            initial_object_vertex_position[1],
            final_object_vertex_position[0],
            final_object_vertex_position[1],
        )
        if delta_angle < 0:
            rotate_direction = "CW"
        else:
            rotate_direction = "CCW"
        delta_distance = Point(pole).distance(Point(final_pole))
        if delta_distance > 0.001:
            moving_direction_temp = [
                (final_pole[0] - pole[0]) / delta_distance,
                (final_pole[1] - pole[1]) / delta_distance,
            ]
            for j in range(len(profile_environment) - 1):
                if (
                    Polygon(initial_object_vertex_position)
                    .buffer(accuracy)
                    .intersects(
                        LineString([profile_environment[j], profile_environment[j + 1]])
                    )
                ):
                    contact_env_edge = [
                        profile_environment[j],
                        profile_environment[j + 1],
                    ]
                    break
            contact_env_edge_direction = [
                profile_environment[j + 1][0] - profile_environment[j][0],
                profile_environment[j + 1][1] - profile_environment[j][1],
            ]
            contact_env_edge_direction = [
                contact_env_edge_direction[0]
                / LineString(
                    [profile_environment[j], profile_environment[j + 1]]
                ).length,
                contact_env_edge_direction[1]
                / LineString(
                    [profile_environment[j], profile_environment[j + 1]]
                ).length,
            ]
            if np.dot(contact_env_edge_direction, moving_direction_temp) > 0:
                moving_direction = contact_env_edge_direction
            else:
                moving_direction = [
                    -contact_env_edge_direction[0],
                    -contact_env_edge_direction[1],
                ]
        if abs(delta_angle) > 1:
            previous_action_total.append("Tip")
            contact_state_trajectory.append(initial_contact_state)
            action_parameter_trajectory.append(
                [pole, rotate_direction, abs(delta_angle)]
            )
            object_pose_trajectory.append(
                Tip_next_vertex_position(
                    initial_object_vertex_position,
                    [pole, rotate_direction, abs(delta_angle)],
                    profile_environment,
                )
            )
        if delta_distance > 0.001:
            previous_action_total.append("Push")
            contact_state_trajectory.append(initial_contact_state)
            action_parameter_trajectory.append([moving_direction, delta_distance])
            object_pose_trajectory.append(final_object_vertex_position)
    elif "double_contact" in initial_contact_state:
        delta_angle = calculate_intersect_angle_AB_CD(
            initial_object_vertex_position[0],
            initial_object_vertex_position[1],
            final_object_vertex_position[0],
            final_object_vertex_position[1],
        )
        previous_action_total.append("Tilting-slide")
        contact_state_trajectory.append(initial_contact_state)
        action_parameter_trajectory.append([initial_contact_state, delta_angle])
        object_pose_trajectory.append(final_object_vertex_position)
    return (
        previous_action_total,
        contact_state_trajectory,
        action_parameter_trajectory,
        object_pose_trajectory,
    )
