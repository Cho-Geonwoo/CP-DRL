# Taken from https://arxiv.org/abs/1910.07224
# Modified by Clément Romac, copy of the license at TeachMyAgent/teachers/LICENSES/ALP-GMM

import numpy as np
from gym.spaces import Box
from collections import deque
import copy
from treelib import Tree
from itertools import islice
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher


def proportional_choice(v, random_state, eps=0.0):
    """
    Return an index of `v` chosen proportionally to values contained in `v`.

    Args:
        v: List of values
        random_state: Random generator
        eps: Epsilon used for an Epsilon-greedy strategy
    """
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]


# A region is a subspace of the task space
class Region(object):
    """
    Subspace of the task space
    """

    def __init__(self, maxlen, r_t_pairs=None, bounds=None, alp=None):
        self.r_t_pairs = r_t_pairs  # A list of pairs of sampled tasks associated to the reward the student obtained
        self.bounds = bounds
        self.alp = alp  # Absolute Learning Progress of Region
        self.maxlen = maxlen

    def add(self, task, reward, is_leaf):
        self.r_t_pairs[1].append(task.copy())
        self.r_t_pairs[0].append(reward)

        need_split = False
        if is_leaf and (len(self.r_t_pairs[0]) > self.maxlen):
            # Leaf is full, need split
            need_split = True
        return need_split


# Implementation of Robust Intelligent-Adaptive-Curiosity (with minor improvements)
class RIAC(AbstractTeacher):
    def __init__(
        self,
        mins,
        maxs,
        seed,
        env_reward_lb,
        env_reward_ub,
        max_region_size=200,
        alp_window_size=None,
        nb_split_attempts=50,
        sampling_in_leaves_only=False,
        min_region_size=None,
        min_dims_range_ratio=1 / 6,
        discard_ratio=1 / 4,
    ):
        """
        Implementation of Robust Intelligent-Adaptive-Curiosity (with minor improvements).

        Args:
            max_region_size: Maximal number of (task, reward) pairs a region can hold before splitting
            alp_window_size: Window size to compute ALP
            nb_split_attempts: Number of attempts to find a valid split
            sampling_in_leaves_only: Whether task sampling uses parent and child regions (False) or only child regions (True)
            min_region_size: (Additional trick n°1) Minimum population required for both children when splitting --> set to 1 to cancel
            min_dims_range_ratio: (Additional trick n°2) Mnimum children region size (compared to initial range of each dimension).
                                     Set min_dims_range_ratio to 1/np.inf to cancel.
            discard_ratio: (Additional trick n°3) If after nb_split_attempts, no split is valid, flush oldest points of parent region.
                              If 1 and 2 are cancelled, this will be canceled since any split will be valid
        """

        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        # Maximal number of (task, reward) pairs a region can hold before splitting
        self.maxlen = max_region_size

        self.alp_window = self.maxlen if alp_window_size is None else alp_window_size

        # Initialize Regions' tree
        self.tree = Tree()
        self.regions_bounds = [Box(self.mins, self.maxs, dtype=np.float32)]
        self.regions_alp = [0.0]
        self.tree.create_node(
            "root",
            "root",
            data=Region(
                maxlen=self.maxlen,
                r_t_pairs=[
                    deque(maxlen=self.maxlen + 1),
                    deque(maxlen=self.maxlen + 1),
                ],
                bounds=self.regions_bounds[-1],
                alp=self.regions_alp[-1],
            ),
        )
        self.nb_dims = len(mins)
        self.nb_split_attempts = nb_split_attempts

        # Whether task sampling uses parent and child regions (False) or only child regions (True)
        self.sampling_in_leaves_only = sampling_in_leaves_only

        # Additional tricks to original RIAC, enforcing splitting rules

        # 1 - Minimum population required for both children when splitting --> set to 1 to cancel
        self.minlen = self.maxlen / 20 if min_region_size is None else min_region_size

        # 2 - minimum children region size (compared to initial range of each dimension)
        # Set min_dims_range_ratio to 1/np.inf to cancel
        self.dims_ranges = self.maxs - self.mins
        self.min_dims_range_ratio = min_dims_range_ratio

        # 3 - If after nb_split_attempts, no split is valid, flush oldest points of parent region
        # If 1- and 2- are canceled, this will be canceled since any split will be valid
        self.discard_ratio = discard_ratio

        # book-keeping
        self.sampled_tasks = []
        self.all_boxes = []
        self.all_alps = []
        self.update_nb = -1
        self.split_iterations = []

        self.hyperparams = locals()

    def compute_alp(self, sub_region):
        if len(sub_region[0]) > 2:
            cp_window = min(
                len(sub_region[0]), self.alp_window
            )  # not completely window
            half = int(cp_window / 2)
            # print(str(cp_window) + 'and' + str(half))
            first_half = np.array(sub_region[0])[-cp_window:-half]
            snd_half = np.array(sub_region[0])[-half:]
            diff = first_half.mean() - snd_half.mean()
            cp = np.abs(diff)
        else:
            cp = 0
        alp = np.abs(cp)
        return alp

    def split(self, nid):
        """
        Try `nb_split_attempts` splits on region corresponding to node <nid>
        """
        # Try nb_split_attempts splits on region corresponding to node <nid>
        reg = self.tree.get_node(nid).data
        best_split_score = 0
        best_bounds = None
        best_sub_regions = None
        is_split = False
        for i in range(self.nb_split_attempts):
            sub_reg1 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
            sub_reg2 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]

            # repeat until the two sub regions contain at least minlen of the mother region
            while len(sub_reg1[0]) < self.minlen or len(sub_reg2[0]) < self.minlen:
                # decide on dimension
                dim = self.random_state.choice(range(self.nb_dims))
                threshold = reg.bounds.sample()[dim]
                bounds1 = Box(reg.bounds.low, reg.bounds.high, dtype=np.float32)
                bounds1.high[dim] = threshold
                bounds2 = Box(reg.bounds.low, reg.bounds.high, dtype=np.float32)
                bounds2.low[dim] = threshold
                bounds = [bounds1, bounds2]
                valid_bounds = True

                if np.any(
                    bounds1.high - bounds1.low
                    < self.dims_ranges * self.min_dims_range_ratio
                ):
                    valid_bounds = False
                if np.any(
                    bounds2.high - bounds2.low
                    < self.dims_ranges * self.min_dims_range_ratio
                ):
                    valid_bounds = valid_bounds and False

                # perform split in sub regions
                sub_reg1 = [
                    deque(maxlen=self.maxlen + 1),
                    deque(maxlen=self.maxlen + 1),
                ]
                sub_reg2 = [
                    deque(maxlen=self.maxlen + 1),
                    deque(maxlen=self.maxlen + 1),
                ]
                for i, task in enumerate(reg.r_t_pairs[1]):
                    if bounds1.contains(task):
                        sub_reg1[1].append(task)
                        sub_reg1[0].append(reg.r_t_pairs[0][i])
                    else:
                        sub_reg2[1].append(task)
                        sub_reg2[0].append(reg.r_t_pairs[0][i])
                sub_regions = [sub_reg1, sub_reg2]

            # compute alp
            alp = [self.compute_alp(sub_reg1), self.compute_alp(sub_reg2)]

            # compute score
            split_score = len(sub_reg1) * len(sub_reg2) * np.abs(alp[0] - alp[1])
            if split_score >= best_split_score and valid_bounds:
                is_split = True
                best_split_score = split_score
                best_sub_regions = sub_regions
                best_bounds = bounds

        if is_split:
            # add new nodes to tree
            for i, (r_t_pairs, bounds) in enumerate(zip(best_sub_regions, best_bounds)):
                self.tree.create_node(
                    identifier=self.tree.size(),
                    parent=nid,
                    data=Region(
                        self.maxlen, r_t_pairs=r_t_pairs, bounds=bounds, alp=alp[i]
                    ),
                )
        else:
            assert len(reg.r_t_pairs[0]) == (self.maxlen + 1)
            reg.r_t_pairs[0] = deque(
                islice(
                    reg.r_t_pairs[0],
                    int(self.maxlen * self.discard_ratio),
                    self.maxlen + 1,
                )
            )
            reg.r_t_pairs[1] = deque(
                islice(
                    reg.r_t_pairs[1],
                    int(self.maxlen * self.discard_ratio),
                    self.maxlen + 1,
                )
            )

        return is_split

    def add_task_reward(self, node, task, reward):
        reg = node.data
        nid = node.identifier
        if reg.bounds.contains(task):  # task falls within region
            self.nodes_to_recompute.append(nid)
            children = self.tree.children(nid)
            for n in children:  # if task in region, task is in one sub-region
                self.add_task_reward(n, task, reward)

            need_split = reg.add(task, reward, children == [])  # COPY ALL MODE
            if need_split:
                self.nodes_to_split.append(nid)

    def episodic_update(self, task, reward, is_success):
        self.update_nb += 1

        # Add new (task, reward) to regions nodes
        self.nodes_to_split = []
        self.nodes_to_recompute = []
        new_split = False
        root = self.tree.get_node("root")
        self.add_task_reward(
            root, task, reward
        )  # Will update self.nodes_to_split if needed
        assert len(self.nodes_to_split) <= 1

        # Split a node if needed
        need_split = len(self.nodes_to_split) == 1
        if need_split:
            new_split = self.split(self.nodes_to_split[0])  # Execute the split
            if new_split:
                # Update list of regions_bounds
                if self.sampling_in_leaves_only:
                    self.regions_bounds = [n.data.bounds for n in self.tree.leaves()]
                else:
                    self.regions_bounds = [n.data.bounds for n in self.tree.all_nodes()]

        # Recompute ALPs of modified nodes
        for nid in self.nodes_to_recompute:
            node = self.tree.get_node(nid)
            reg = node.data
            reg.alp = self.compute_alp(reg.r_t_pairs)

        # Collect regions data (regions' ALP and regions' (task, reward) pairs)
        all_nodes = (
            self.tree.all_nodes()
            if not self.sampling_in_leaves_only
            else self.tree.leaves()
        )
        self.regions_alp = []
        self.r_t_pairs = []
        for n in all_nodes:
            self.regions_alp.append(n.data.alp)
            self.r_t_pairs.append(n.data.r_t_pairs)

        # Book-keeping
        if new_split:
            self.all_boxes.append(copy.copy(self.regions_bounds))
            self.all_alps.append(copy.copy(self.regions_alp))
            self.split_iterations.append(self.update_nb)
        assert len(self.regions_alp) == len(self.regions_bounds)

        return new_split, None

    def sample_random_task(self):
        return self.regions_bounds[0].sample()  # First region is root region

    def sample_task(self):
        mode = self.random_state.rand()
        if (
            mode < 0.1
        ):  # "mode 3" (10%) -> sample on regions and then mutate lowest-performing task in region
            if len(self.sampled_tasks) == 0:
                self.sampled_tasks.append(self.sample_random_task())
            else:
                self.sampled_tasks.append(self.non_exploratory_task_sampling()["task"])

        elif mode < 0.3:  # "mode 2" (20%) -> random task
            self.sampled_tasks.append(self.sample_random_task())

        else:  # "mode 1" (70%) -> proportional sampling on regions based on ALP and then random task in selected region
            region_id = proportional_choice(
                self.regions_alp, self.random_state, eps=0.0
            )
            self.sampled_tasks.append(self.regions_bounds[region_id].sample())

        return self.sampled_tasks[-1].astype(np.float32)

    def non_exploratory_task_sampling(self):
        # 1 - Sample region proportionally to its ALP
        region_id = proportional_choice(self.regions_alp, self.random_state, eps=0.0)

        # 2 - Retrieve (task, reward) pair with lowest reward
        worst_task_idx = np.argmin(self.r_t_pairs[region_id][0])

        # 3 - Mutate task by a small amount (using Gaussian centered on task, with 0.1 std)
        task = self.random_state.normal(
            self.r_t_pairs[region_id][1][worst_task_idx].copy(), 0.1
        )
        # clip to stay within region (add small epsilon to avoid falling in multiple regions)
        task = np.clip(
            task,
            self.regions_bounds[region_id].low + 1e-5,
            self.regions_bounds[region_id].high - 1e-5,
        )
        return {
            "task": task,
            "infos": {"bk_index": len(self.all_boxes) - 1, "task_infos": region_id},
        }

    def dump(self, dump_dict):
        dump_dict["all_boxes"] = self.all_boxes
        dump_dict["split_iterations"] = self.split_iterations
        dump_dict["all_alps"] = self.all_alps
        # dump_dict['riac_params'] = self.hyperparams
        return dump_dict

    @property
    def nb_regions(self):
        return len(self.regions_bounds)

    @property
    def get_regions(self):
        return self.regions_bounds
