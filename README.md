# LDHP

**Library-Driven Hierarchical Planning for Non-Prehensile Dexterous Manipulation**

This repository provides an implementation of *LDHP*, a framework that autonomously
synthesizes executable manipulation plans for non-prehensile, contact-rich tasks.
By jointly reasoning over a contact-state planner, a grasp-sequence planner, and a
primitive library, LDHP eliminates the need for expert-designed, task-specific
control strategies.

## Inputs

The planner requires the following inputs:

- a description of the environment (e.g., supporting surfaces, slots, and fixtures);
- the initial pose and opening of the parallel-jaw gripper;
- the shape, dimensions, and density distribution of the manipulated object;
- the initial and target object poses;
- the kinematic and geometric model of the gripper to be used; and
- the friction coefficient between the object and its surroundings.

## Examples

The repository includes four representative tasks illustrating the framework:

1. **Task 1** — Lifting an object off a flat surface with a passive gripper of
   zero mobility. Run `task1.py`.
2. **Task 2** — Placing an object onto a flat surface with a passive gripper of
   zero mobility. Run `task2.py`.
3. **Task 3** — Scooping an object out of a slot. Run `task3.py`.
4. **Task 4** — Inserting an object into a slot. Run `task4.py`.

## Dependencies

- Python 3.10
- Gurobipy (install via `pip install gurobipy`; a valid Gurobi license is required)
