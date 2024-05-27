#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import torch
import copy
import os
import pickle
import random
import itertools

import proplogic
import formulae


class PropFormulaeDataset:
    def __init__(self, leaf_prob, max_depth, n_vars, n_graphs, max_nodes, atomic_pred, save_path):
        self.leaf_prob = leaf_prob
        self.max_depth = max_depth
        self.n_vars = n_vars
        self.n_graphs = n_graphs
        self.max_nodes = max_nodes
        self.sampler = formulae.PropFormula(leaf_prob=self.leaf_prob, inner_node_prob=None, max_depth=self.max_depth)
        self.atomic_pred = atomic_pred
        self.dataset_folder = save_path

    @property
    def _n_graphs(self):
        return self.n_graphs

    @_n_graphs.setter
    def _n_graphs(self, ng):
        self.n_graphs = ng

    @staticmethod
    def get_id(child_name, name, label_dict, idx):
        while child_name in label_dict:
            idx += 1
            child_name = name + "(" + str(idx) + ")"
        return child_name, idx

    @staticmethod
    def traverse(formula, idx, label_dict):
        # DFS traversal of the tree
        current_node = formula
        current_node_name = current_node.name + "(" + str(idx) + ")"
        label_dict[current_node_name] = current_node.name
        edges = []
        if type(current_node) is not proplogic.LogicVar:
            left_child = current_node.left_child if type(current_node) is not proplogic.Negation else current_node.child
            current_idx = idx + 1
            left_child_name = left_child.name + "(" + str(current_idx) + ")"
            left_child_name, current_idx = PropFormulaeDataset.get_id(left_child_name, left_child.name, label_dict,
                                                                      current_idx)
            edges.append([current_node_name, left_child_name])
            e, d = PropFormulaeDataset.traverse(left_child, current_idx, label_dict)
            edges += e
            label_dict.update(d)
            if type(current_node) is not proplogic.Negation:
                right_child = current_node.right_child
                current_idx = idx + 2
                right_child_name = right_child.name + "(" + str(current_idx) + ")"
                right_child_name, current_idx = PropFormulaeDataset.get_id(right_child_name, right_child.name,
                                                                           label_dict, current_idx)
                edges.append([current_node_name, right_child_name])
                e, d = PropFormulaeDataset.traverse(right_child, current_idx, label_dict)
                edges += e
                label_dict.update(d)
        return edges, label_dict

    @staticmethod
    def build_tree(formula):
        edges, label_dict = PropFormulaeDataset.traverse(formula, 0, {})
        graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
        assert (nx.is_tree(graph))
        return graph, label_dict

    @staticmethod
    def get_simple_attr(attr, n_operators=3):
        simple_attr = torch.zeros((attr.shape[0], n_operators + 1))
        simple_attr[:, :-1] = attr[:, :n_operators]  # .cpu()
        simple_attr[torch.where(~torch.any(simple_attr, dim=1))[0], n_operators] = 1
        return simple_attr

    @staticmethod
    def get_matrices(graph, n_vars, n_operators=3, simplified=False):
        def find_index(d, s): return [d[ss] if ss in d else n_operators + int(ss.partition('_')[2].partition(' ')[0])
                                      for ss in s]

        def get_name(n): return n.partition('(')[0]
        adj = torch.from_numpy(nx.to_numpy_array(graph))
        one_hot_dict_idx = {'and': 0, 'or': 1, 'not': 2}  # TODO: add flexibility for more operators
        names = list(map(get_name, graph.nodes()))
        part_names = [n.partition('_')[2].partition(' ')[2].partition(' ') for n in names]
        idx_list = find_index(one_hot_dict_idx, names)
        threshold_list = torch.Tensor([float(n[2]) if n[2] else 0 for n in part_names])
        sign_dict = {'<=': n_operators + n_vars, '>=': n_operators + n_vars + 1, '': 0}
        threshold_idx = [sign_dict[n[0]] for n in part_names]
        thresh = True if max(threshold_idx) > 0 else False
        dim_attr = n_operators + n_vars + 2 if thresh else n_operators + n_vars
        attr = torch.zeros((adj.shape[0], dim_attr))
        if thresh:
            attr[torch.arange(attr.shape[0]), threshold_idx] = threshold_list
        attr[torch.arange(attr.shape[0]), idx_list] = 1
        if simplified:
            simple_attr = PropFormulaeDataset.get_simple_attr(attr, n_operators=n_operators)
            return adj, attr, simple_attr
        return adj, attr

    @staticmethod
    def get_input(input_formula, n_vars, simplified=False):
        g, _ = PropFormulaeDataset.build_tree(input_formula)
        if simplified:
            return PropFormulaeDataset.get_matrices(g, n_vars, simplified=True)
        return PropFormulaeDataset.get_matrices(g, n_vars, simplified=False)

    def generate_dataset(self, file_name, simplified=False, augment_pred=True, repeats=10, save=True):
        graphs = []
        while len(graphs) < self.n_graphs:
            if self.atomic_pred and not augment_pred:
                self.sampler.atomic_pred = True
            phi = self.sampler.sample(self.n_vars)
            g = PropFormulaeDataset.get_input(phi, self.n_vars, simplified=simplified)
            if g[0].shape[0] < self.max_nodes:
                graphs.append(g)
                if augment_pred:
                    assert (self.atomic_pred is True)
                    for _ in range(repeats):
                        graphs.append(make_vars_atomic_preds(g))
        random.shuffle(graphs)
        if save:
            with open(self.dataset_folder + os.path.sep + file_name, 'wb') as f:
                pickle.dump(graphs, f)
        return graphs

    @staticmethod
    def get_data_statistics(dataset, file_name, save=True):
        # statistic about number of variables, number of nodes, depth of the tree
        def get_n_vars(attr): return attr.shape[1] - 3  # operators
        def get_n_nodes(adj): return adj.shape[0]
        def get_depth(adj): return len(nx.dag_longest_path(G=nx.from_numpy_matrix(adj.cpu().numpy(),
                                                                                  create_using=nx.DiGraph)))
        if len(list(zip(*dataset))) == 3:
            adjs, attrs, simple_attrs = list(zip(*dataset))
        else:
            adjs, attrs = list(zip(*dataset))
        n_vars, n_nodes, depths = [list(map(get_n_vars, attrs)), list(map(get_n_nodes, adjs)),
                                   list(map(get_depth, adjs))]
        n_vars, n_nodes, depths = [torch.Tensor(i) for i in [n_vars, n_nodes, depths]]
        stats = {'n_vars': [torch.min(n_vars), torch.max(n_vars), torch.mean(n_vars), torch.std(n_vars)],
                 'n_nodes': [torch.min(n_nodes), torch.max(n_nodes), torch.mean(n_nodes), torch.std(n_nodes)],
                 'depths': [torch.min(depths), torch.max(depths), torch.mean(depths), torch.std(depths)]}
        if save:
            with open(file_name, 'wb') as f:
                pickle.dump(stats, f)
        return stats


class PropFormulaeLoader:
    def __init__(self, leaf_prob, max_depth, n_vars, n_graphs, max_nodes, atomic_pred, device, save_path):
        self.leaf_prob = leaf_prob
        self.max_depth = max_depth
        self.n_vars = n_vars
        self.n_graphs = n_graphs
        self.max_nodes = max_nodes
        self.dataset_folder = save_path
        self.device = device
        self.generator = PropFormulaeDataset(leaf_prob, max_depth, n_vars, n_graphs, max_nodes, atomic_pred, save_path)
        self.rep_loader = PropFormulaeIsoLoader(leaf_prob, max_depth, self.n_vars, n_graphs, max_nodes, device,
                                                atomic_pred, save_path)
        self.atomic_pred = atomic_pred

    def get_data(self, simplified=False, kind=None, augment_pred=True, repeats=10, save=True, dvae=True, min_depth=0):
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        files = os.listdir(self.dataset_folder)
        train_list, test_list, val_list = [[i for i in files if i.startswith(j)] for j in
                                           ['training', 'test', 'validation']]
        if kind == 'train' or kind == 'all':
            if train_list:
                with open(self.dataset_folder + os.path.sep + train_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                # generate name
                train_name = 'training_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                val_name = 'validation_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                test_name = 'test_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                all_data = self.rep_loader.get_batch_representatives(self.n_vars, augment_pred=augment_pred,
                                                                     repeats=repeats, simplified=simplified,
                                                                     min_depth=min_depth)
                random.shuffle(all_data)
                if kind == 'all':
                    attr_idx = 2 if (simplified and len(all_data[0]) > 2) else 1
                    all_data = [[d[0].to(self.device), d[attr_idx].to(self.device)] for d in all_data]
                    return all_data
                train_end_idx = int(0.80*len(all_data))
                val_end_idx = int(0.90*len(all_data)) if (int(0.90*len(all_data)) - train_end_idx) < 500 \
                    else train_end_idx + 500
                data = all_data[:train_end_idx]
                validation_data = all_data[train_end_idx+1:val_end_idx]
                test_data = all_data[val_end_idx+1:]
                if save:
                    with open(self.dataset_folder + os.path.sep + train_name, 'wb') as f:
                        pickle.dump(data, f)
                with open(self.dataset_folder + os.path.sep + val_name, 'wb') as f:
                    pickle.dump(validation_data, f)
                with open(self.dataset_folder + os.path.sep + test_name, 'wb') as f:
                    pickle.dump(test_data, f)
                self.generator.get_data_statistics(data, self.dataset_folder + os.path.sep +
                                                   'stat_training_p={}_max-depth={}.pickle'.format(self.leaf_prob,
                                                                                                   self.max_depth))
                self.generator.get_data_statistics(validation_data, self.dataset_folder + os.path.sep +
                                                   'stat_validation_p={}_max-depth={}.pickle'.format(self.leaf_prob,
                                                                                                     self.max_depth))
                self.generator.get_data_statistics(test_data, self.dataset_folder + os.path.sep +
                                                   'stat_test_p={}_max-depth={}.pickle'.format(self.leaf_prob,
                                                                                               self.max_depth))
        elif kind == 'test':
            if test_list:
                with open(self.dataset_folder + test_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                self.generator.n_graphs = 100
                # generate name
                name = 'test_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                data = self.generator.generate_dataset(name, simplified=simplified, save=save)
                self.generator.get_data_statistics(data, self.dataset_folder + os.path.sep +
                                                   'stat_test_p={}_max-depth={}.pickle'.format(self.leaf_prob,
                                                                                               self.max_depth))
        elif kind == 'validation':
            if val_list:
                with open(self.dataset_folder + os.path.sep + val_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                ng = 100
                self.generator.n_graphs = ng
                # generate name
                name = 'validation_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                data = self.generator.generate_dataset(name, simplified=simplified, save=save)
                self.generator.get_data_statistics(data, self.dataset_folder + os.path.sep +
                                                   'stat_validation_p={}_max-depth={}.pickle'.format(self.leaf_prob,
                                                                                                     self.max_depth))
        attr_idx = 2 if (simplified and len(data[0]) > 2) else 1
        data = [[d[0].to(self.device), d[attr_idx].to(self.device)] for d in data]
        if dvae and kind in ['test', 'validation']:
            data = PropFormulaeLoader.get_dvae_input(data)
        return data

    @staticmethod
    def get_dvae_input(data):
        dvae_data = []
        for d in data:
            adj, attr = [d[0], d[1]]
            n_types = attr.shape[1] + 2  # need to add start node and end node
            root_row_idx = torch.where(~torch.any(adj, dim=0))[0]
            if len(root_row_idx) > 1:  # unique root node
                root_row_idx = root_row_idx[0]
            leaf_row_idx = torch.where(~torch.any(adj, dim=1))[0] + 1  # + 1 to take into account start node index
            # adjust adjacency matrix
            new_adj = torch.zeros((adj.shape[0] + 2, adj.shape[1] + 2))  # start and end nodes
            new_adj[1:-1, 1:-1] = torch.clone(adj)  # start node is first row, end node is last row
            new_adj[0, root_row_idx + 1] = 1
            new_adj[leaf_row_idx, -1] = 1  # leaves are connected to end node
            # adjust feature matrix
            new_attr = torch.zeros((new_adj.shape[0], n_types))
            new_attr[1:-1, 2:] = torch.clone(attr)  # 0-th type is start node, 1-th type is end node
            new_attr[0, 0] = 1  # 0-th type is start type (row 0)
            new_attr[-1, 1] = 1  # 1-th type is end type  (last row)
            # simplify to get the actual number of variables
            # TODO: add flexibility for number of operators
            x = torch.Tensor([True] * 5).bool()
            y = new_attr.sum(dim=0).bool()[5:]
            cols = torch.cat([x, y])
            new_attr = new_attr[:, cols]
            dvae_data.append([new_adj, new_attr])
        return dvae_data

    @staticmethod
    def divide_batches(dataset, batch_size, n_data):
        batches = []
        for b in range(len(dataset) // batch_size):
            batches.append(dataset[b * batch_size:b * batch_size + batch_size])
        last_idx = (len(dataset) // batch_size - 1) * batch_size + batch_size
        if last_idx < n_data - 1:
            batches.append(dataset[last_idx:])
        return batches

    def load_batches(self, arg, save=True, dvae=True, min_depth=0):
        # TODO: add new loading option (or maybe it should be placed in the main)
        # TODO: remove things related to the change of training set at each checkpoint
        # TODO: decide how to deal with initial embeddings (especially if they are not there)!
        #  maybe a totally new function should be though (in the main directly maybe)
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        dataset = self.get_data(simplified=(not arg.var_indexes), kind='train', augment_pred=arg.augment_pred,
                                repeats=arg.repeats, save=save, min_depth=min_depth)
        files = os.listdir(self.dataset_folder)
        dataset_prefix = [f for f in files if f.startswith('train')]
        dataset_name = dataset_prefix[0] if len(dataset_prefix) > 0 else 'train_current'
        n_data = len(dataset)
        stat_name = 'stat_training' + dataset_name[13:] \
            if (len(dataset_name) > 13 and dataset_name[:13] == 'training_data') else 'stat_' + dataset_name
        if not os.path.isfile(self.dataset_folder + os.path.sep + stat_name) and stat_name != 'stat_train_current':
            self.generator.get_data_statistics(dataset, self.dataset_folder + os.path.sep + stat_name)
        if dvae:
            dataset = PropFormulaeLoader.get_dvae_input(dataset)
        batches = PropFormulaeLoader.divide_batches(dataset, arg.batch_size, n_data)
        return batches, stat_name, n_data


class PropFormulaeIsoLoader:
    def __init__(self, leaf_prob, max_depth, n_vars, n_batches, max_nodes, device, atomic_pred, save_path):
        self.leaf_prob = leaf_prob
        self.max_depth = max_depth
        self.n_vars = n_vars
        self.n_batches = n_batches
        self.max_nodes = max_nodes
        self.dataset_folder = save_path
        self.device = device
        self.generator = PropFormulaeDataset(leaf_prob, max_depth, n_vars, n_batches, max_nodes, atomic_pred, save_path)
        self.atomic_pred = atomic_pred

    def get_batch_representatives(self, n_vars, augment_pred=True, repeats=10, simplified=False, min_depth=0):
        n_representatives = 0
        g_batch = []
        rep_str = set()
        while n_representatives < self.n_batches:
            if self.atomic_pred and not augment_pred:
                self.generator.sampler.atomic_pred = True
            else:
                self.generator.sampler.atomic_pred = False
            g_phi = self.generator.sampler.sample(n_vars)
            g_depth = g_phi.depth()
            g_rep_mat = self.generator.get_input(g_phi, n_vars, simplified=simplified)
            if g_rep_mat[0].shape[0] > self.max_nodes:
                continue
            g_adj, g_attr = [g_rep_mat[0], g_rep_mat[2]] if simplified else [g_rep_mat[0], g_rep_mat[1]]
            if g_depth <= min_depth or g_rep_mat[1].shape[1] <= 3 + 2:
                continue
            rep_str.add(str(from_output_to_formula(g_adj, g_attr)))
            if len(rep_str) > n_representatives and augment_pred:
                n_representatives += 1
                if augment_pred is True:
                    assert (self.atomic_pred is True)
                    for _ in range(repeats):
                        g_new_phi = make_vars_atomic_preds(g_phi)
                        g_new_adj, g_new_attr = self.generator.get_input(g_new_phi, n_vars, simplified=simplified)
                        g_batch.append([g_new_adj, g_new_attr])
                else:
                    g_batch.append([g_adj, g_attr])
                    n_representatives += 1
        return g_batch

    @staticmethod
    def find_index_sym_operators(adj, attr, simplified=True):
        leaves_idx = np.where(~adj.any(axis=1))[0]
        assert np.all((leaves_idx == torch.where(~torch.any(adj, dim=1))[0].numpy()))
        and_or_idx = np.sort(np.concatenate((np.where(attr[:, 0] == 1)[0],
                                             np.where(attr[:, 1] == 1)[0]), axis=None))[::-1]
        sym_op_dict = {}
        for idx in and_or_idx:
            if simplified:
                if len(np.intersect1d(np.where(adj[idx, :] != 0)[0], leaves_idx)) < 2:
                    sym_op_dict[idx] = np.where(adj[idx, :] != 0)[0]
            else:
                if len(np.intersect1d(np.where(adj[idx, :] != 0)[0], leaves_idx)) == 2:
                    children_idx = np.where(adj[idx, :] != 0)[0]
                    children_var_idx = set([np.where(attr[child_idx, :] != 0)[0][0] for child_idx in children_idx])
                    if len(children_var_idx) > 1:
                        sym_op_dict[idx] = np.where(adj[idx, :] != 0)[0]
                else:
                    sym_op_dict[idx] = np.where(adj[idx, :] != 0)[0]
        return sym_op_dict

    @staticmethod
    def find_last_child_idx(adj, parent_idx):
        max_child = torch.max(torch.nonzero(adj[parent_idx, :]).flatten()) if (torch.nonzero(
            adj[parent_idx, :]).flatten()).shape[0] > 0 else None
        # np.max(np.where(adj[parent_idx, :] != 0)[0]) if len(np.where(adj[parent_idx, :] != 0)[0]) > 0 \
        #     else None
        current_max = -1
        while current_max != 0 and max_child:
            # if len(np.where(adj[max_child, :] != 0)[0]) > 0:
            if torch.nonzero(adj[max_child, :]).flatten().shape[0] > 0:
                current_max = torch.max(torch.nonzero(adj[max_child, :]).flatten())
                # np.max(np.where(adj[max_child, :] != 0)[0])
            else:
                current_max = 0
            if current_max > max_child:
                max_child = current_max
        return max_child

    @staticmethod
    def find_first_child_idx(adj, parent_idx):
        min_child = torch.min(torch.nonzero(adj[parent_idx, :]).flatten()) if torch.nonzero(
            adj[parent_idx, :]).flatten().shape[0] > 0 else None
        # np.min(np.where(adj[parent_idx, :] != 0)[0]) if len(np.where(adj[parent_idx, :] != 0)[0]) > 0 \
        #    else None
        current_max = -1
        while current_max != 0 and min_child:
            # if len(np.where(adj[min_child, :] != 0)[0]) > 0:
            if torch.nonzero(adj[min_child, :]).flatten().shape[0] > 0:
                current_max = torch.max(torch.nonzero(adj[min_child, :]).flatten())
                # np.max(np.where(adj[min_child, :] != 0)[0])
            else:
                current_max = 0
            if current_max > min_child:
                min_child = current_max
        return min_child

    @staticmethod
    def swap_subtree(adj, attr, parent_idx, op_dict):
        child_start = op_dict[parent_idx]
        first_idx = [child_start[0], child_start[1]]
        second_idx = [child_start[1], PropFormulaeIsoLoader.find_last_child_idx(adj, parent_idx)]
        # subset of rows already swapped
        swapped_block_adj_row_part = np.concatenate((copy.deepcopy(adj[second_idx[0]:second_idx[1] + 1, :]),
                                                     copy.deepcopy(adj[first_idx[0]:first_idx[1], :])), axis=0)
        # also parent row (because of swapping of the columns)
        swapped_block_adj_row_part = np.concatenate((copy.deepcopy(adj[parent_idx, :].unsqueeze(0)),
                                                     swapped_block_adj_row_part), axis=0)
        # column corresponding to right most child (to be inserted in place of the left most)
        col_second_block = copy.deepcopy(swapped_block_adj_row_part[:, second_idx[0]:second_idx[1] + 1])
        # remove columns corresponding to right most child
        swapped_block_adj_row = np.delete(copy.deepcopy(swapped_block_adj_row_part),
                                          list(np.arange(second_idx[0], second_idx[1] + 1)), axis=1)
        # insert them in the right place
        swapped_block_adj = np.insert(swapped_block_adj_row, first_idx[0], col_second_block.transpose(), axis=1)
        swapped_block_attr = np.concatenate((copy.deepcopy(attr[second_idx[0]:second_idx[1] + 1, :]),
                                             copy.deepcopy(attr[first_idx[0]:first_idx[1], :])), axis=0)
        new_adj = np.delete(copy.deepcopy(adj), [parent_idx] + list(np.arange(first_idx[0], second_idx[1] + 1)), axis=0)
        new_attr = np.delete(copy.deepcopy(attr), list(np.arange(first_idx[0], second_idx[1] + 1)), axis=0)
        new_adj = np.insert(new_adj, parent_idx, swapped_block_adj[0, :], axis=0)
        new_adj = np.insert(new_adj, first_idx[0], swapped_block_adj[1:, :], axis=0)
        new_attr = np.insert(new_attr, first_idx[0], swapped_block_attr, axis=0)
        return new_adj, new_attr

    @staticmethod
    def generate_all_permutations_no_var(adj, attr, b_size, device, simplified=True):
        # TODO: add var option (how?)
        sym_op_dict = PropFormulaeIsoLoader.find_index_sym_operators(adj, attr, simplified=simplified)
        n_operators = len(sym_op_dict)
        tot_perm = []
        for n_swaps in range(1, n_operators + 1):
            tot_perm += list(itertools.combinations(np.arange(n_operators), n_swaps))
        n_perm = 2 ** n_operators
        perm = tot_perm if n_perm <= 5 * b_size else [tot_perm[i] for i in
                                                      np.random.choice(len(tot_perm), size=5 * b_size)]
        perm_graph = []
        for p in perm:  # indexes to swap
            new_adj, new_attr = [copy.deepcopy(adj), copy.deepcopy(attr)]
            for p_idx in list(p):
                new_adj, new_attr = PropFormulaeIsoLoader.swap_subtree(new_adj, new_attr,
                                                                       list(sym_op_dict.keys())[p_idx], sym_op_dict)
            perm_graph.append([new_adj.to(device), new_attr.to(device)])
        perm_graph.append([adj.to(device), attr.to(device)])
        return perm_graph

    def get_iso_batches(self, b_size, augment_pred=True, repeats=10, simplified=False, save=True):
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        files = os.listdir(self.dataset_folder)
        train_batch_list = [i for i in files if i.startswith('batch_train')]
        if train_batch_list:
            with open(self.dataset_folder + os.path.sep + train_batch_list[0], 'rb') as f:
                data = pickle.load(f)
        else:
            file_name = 'batch_training_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
            data, rep = [[], []]  # rep is used for doing statistics in the dataset
            for n_vars in self.n_vars:
                var_rep = self.get_batch_representatives(n_vars, augment_pred=augment_pred, repeats=repeats,
                                                         simplified=simplified)
                rep += var_rep
            random.shuffle(rep)
            for g_rep in rep:
                current_batch = self.generate_all_permutations_no_var(g_rep[0], g_rep[1], b_size, self.device,
                                                                      simplified)
                if len(current_batch) <= b_size:
                    data.append(current_batch)
                else:
                    for b in range(len(current_batch) // b_size):
                        data.append(current_batch[b * b_size:b * b_size + b_size])
                    last_idx = (len(current_batch) // b_size - 1) * b_size + b_size
                    if last_idx < len(current_batch) - 1:
                        data.append(current_batch[last_idx:])
            stat_name = self.dataset_folder + os.path.sep + 'stat_batch_training_p={}_max-depth={}.pickle'.format(
                self.leaf_prob, self.max_depth)
            self.generator.get_data_statistics(rep, stat_name)
            if save:
                with open(self.dataset_folder + os.path.sep + file_name, 'wb') as f:
                    pickle.dump(data, f)
        return data

    def load_batches(self, arg, save=True, dvae=True):
        # TODO: assume to load the entire dataset
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        batch_dataset = self.get_iso_batches(arg.batch_size, augment_pred=arg.augment_pred, repeats=arg.repeats,
                                             simplified=(not arg.var_indexes), save=save)
        files = os.listdir(self.dataset_folder)
        dataset_name = [f for f in files if f.startswith('batch_train')][0]
        stat_name = 'stat_batch_training' + dataset_name[14:] \
            if (len(dataset_name) > 14 and dataset_name[:14] == 'batch_training') else 'stat_batch_' + dataset_name
        if not os.path.isfile(self.dataset_folder + os.path.sep + stat_name):
            rep = [b[0] for b in batch_dataset]
            self.generator.get_data_statistics(rep, self.dataset_folder + os.path.sep + stat_name)
        n_data = sum([len(batch) for batch in batch_dataset])
        if dvae:
            dvae_batches = [PropFormulaeLoader.get_dvae_input(batch) for batch in batch_dataset]
            batch_dataset = dvae_batches
        return batch_dataset, stat_name, n_data