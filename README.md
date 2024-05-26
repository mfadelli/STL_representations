# STL_representations
All files except the file ``stl_dataset.py`` have only been modified by adding comments (useful to me) and should operate in the same way as the original files.

## Representations implemented
### Parameters
To construct the graph I implemented four parameters: ``variant``, ``shared_var``, ``num_arg`` and ``edge_feats``.
- ``variant`` has three possible choices: *original*, *threshold-sign* and *all-in-var*.
  - *original* is the version implemented in the original code, leaf node ----> placeholder-variable-sign-threshold.
  - *threshold-sign* leaf node ----> placeholder-threshold/sign-variable.
  - *all-in-var* leaf node ----> placeholder-variable.
- ``shared_var`` is a Boolean parameter.
  - if *True* we have a single node for each variable index (e.g. x_1) as in the original code.
  - if *False* we have a tree-like structure with each instance of the variable as a separate node (e.g. x_1(8), x_1(11)).
- ``num_arg`` is a Boolean parameter.
  - if *True* the edge between the two arguments of the Until operator is not present and we add a binary slot for each node which is 1 if and only if the node is the right child of an until node.
  - if *False* the number of the argument is encoded by an edge from the first children to the second as in the original code.
- ``edge_feats`` introduce the edge features. It has two modes plus *False* if edge features are not needed.
  - in *single* mode the edge features are [0,0] for all the edges except the ones from the temporal operators that are [left_bound, right_bound] of the temporal interval. Temporal intervals are still encoded in the node features.
  - *cumulative* follows the idea that intervals are summed for temporal operators and kept constant for non-temporal operator. Starts from [0,0].
  
### Shortcuts for some representations 
Morover some function have the parameter ``representation`` that allows to set all these parameters by passing the name of the representations. At the moment the following representations are implemented by name:
- ``T1`` with params [variant=all-in-var, shared_var=False, num_arg=True, edge_feats=False].
- ``T2`` with params [variant=all-in-var, shared_var=False, num_arg=True, edge_feats=single].
- ``DAG1`` with params [variant=all-in-var, shared_var=False, num_arg=False, edge_feats=False].
- ``DAG2`` with params [variant=all-in-var, shared_var=True, num_arg=True, edge_feats=single].
- ``original`` with params [variant=original, shared_var=True, num_arg=False, edge_feats=False] (the one present in the original code).

They can easily be changed manually, especially with respect to the ``variant`` parameter to represent the leaf nodes. (No guarantee that they will work)

## Things to be fixed
- I'm not sure that anything works set aside for some small examples. I need to conduct more tests with different parameters.
- I'm not sure if some parameters may cause each other problems (not all combinations have been tried).
- Possibility to add argument number as edge feature (but maybe not very easy to code given the current structure).
- Code for edge features now rely on the assumption that each node has only one parent. This is not true for many representation. Problems arise especially with the combination of ``cumulative`` and the presence of an edge for argument number (also a problem in theory, what value should be assigned to that fictitious edge in the cumulative setting?).
- The edge feature matrix is a list[list[list]]. I have to find a more pytorch-ic way to store it. It is also a sparse matrix (as the adjacency one).
- My code is very ugly.

## Some details of the added functions
- ``rep_to_pars`` : shortcut to set representation parameters by passing the name of the representation. See above.
- ``get_edge_attr ``: generates the edge features "matrix" given the other matrices. It calls ``generate_edge_features`` for computing the geatures of an edge.
- ``add_leaf_child`` is the most changed function to allow other leaf representations. Also ``traverse_formula`` and ``get_matrices`` were modified.


