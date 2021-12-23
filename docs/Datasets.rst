Datasets
==================

Graphormer supports training with both existing datasets in graph libraries and customized datasets. 

Existing Datasets
~~~~~~~~~~~~~~~~~

Graphormer supports training with datasets in existing libraries.
Users can easily exploit datasets in these libraries by specifying the ``--dataset-source`` and ``--dataset-name`` parameters.

``--dataset-source`` specifies the source for the dataset, can be:

1. ``dgl`` for `DGL <https://docs.dgl.ai/>`__

2. ``pyg`` for `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`__

3. ``ogb`` for `OGB <https://ogb.stanford.edu/>`__

``--dataset-name`` specifies the dataset in the source.
For example, by specifying ``--dataset-source pyg`` and ``--dataset-name zinc``, Graphormer will load the `ZINC <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC>`__ dataset from Pytorch Geometric.
When a dataset requires additional parameters to construct, the parameters are specified as ``<dataset_name>:<param_1>=<value_1>,<param_2>=<value_2>,...,<param_n>=<value_n>``.
When the type of a parameter value is a list, the value is represented as a string with the list elements concatenated by `+`.
For example, if we want to specify multiple ``label_keys`` with ``mu``, ``alpha``, and ``homo`` for `QM9 <https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm9-dataset>`__ dataset,
``--dataset-name`` should be ``qm9:label_keys=mu+alpha+homo``.

When dataset split (``train``, ``valid`` and ``test`` subsets) is not configured in the original dataset source, we randomly partition
the full set into ``train``, ``valid`` and ``test`` with ratios ``0.7``, ``0.2`` and ``0.1``, respectively.
If you want customized split of a dataset, you may implement a `customized dataset `.
Currently, only integer features of nodes and edges in the datasets are used.

A full list of supported datasets of each data source:

+------------------+----------------+-----------------------------------------+-----------------------------+
| Dataset Source   | Dataset Name   | Link                                    | #Label/#Class               |
+==================+================+=========================================+=============================+
| ``dgl``          |   ``qm7b``     | QM7B_ dataset                           |      14                     |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |   ``qm9``      | QM9_  dataset                           | Depending on ``label_keys`` |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |   ``qm9edge``  | QM9Edge_ dataset                        | Depending on ``label_keys`` |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |   ``minigc``   | MiniGC_ dataset                         |        8                    |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |   ``gin``      | `Graph Isomorphism Network`_ dataset    |        1                    |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  | ``fakenews``   | `FakeNewsDataset`_ dataset              |        1                    |
+------------------+----------------+-----------------------------------------+-----------------------------+
| ``pgy``          |``moleculenet`` | MoleculeNet_ dataset                    |             1               |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |   ``zinc``     | ZINC_ dataset                           |             1               |
+------------------+----------------+-----------------------------------------+-----------------------------+
| ``ogb``          |``ogbg-molhiv`` | ogbg-molhiv_ dataset                    |             1               |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |``ogbg-molpcba``| ogbg-molpcba_ dataset                   |             128             |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |``pcqm4m``      | PCQM4M_ dataset                         |               1             |
|                  +----------------+-----------------------------------------+-----------------------------+
|                  |``pcqm4mv2``    | PCQM4Mv2_ dataset                       |               1             |
+------------------+----------------+-----------------------------------------+-----------------------------+


.. _QM7B: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm7b-dataset
.. _QM9: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm9-dataset
.. _QM9Edge: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm9edge-dataset
.. _MiniGC: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#mini-graph-classification-dataset
.. _TU: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#tu-dataset
.. _Graph Isomorphism Network: https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm9-dataset
.. _FakeNewsDataset: https://docs.dgl.ai/en/0.7.x/_modules/dgl/data/fakenews.html

.. _KarateClub: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.KarateClub
.. _MoleculeNet: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.MoleculeNet
.. _ZINC: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC
.. _MD17: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.MD17

.. _ogbg-molhiv: https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
.. _ogbg-molpcba: https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
.. _PCQM4M: https://ogb.stanford.edu/kddcup2021/pcqm4m/
.. _PCQM4Mv2: https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
.. _ogbg-ppa: https://ogb.stanford.edu/docs/graphprop/#ogbg-ppa

.. _Customized Datasets:
Customized Datasets
~~~~~~~~~~~~~~~~~~~

Users may create their own datasets. To use customized dataset:

1. Create a folder (for example, with name `customized_dataset`), and a python script with arbitrary name in the folder.

2. In the created python script, define a function which returns the created dataset. And register the function with ``register_dataset``. Here is a sample python script.
We define a `QM9 <https://docs.dgl.ai/en/0.6.x/api/python/dgl.data.html#qm9-dataset>`__ dataset from ``dgl`` with customized split.

.. code-block:: python
   :linenos:

    from graphormer.data import register_dataset
    from dgl.data import QM9
    import numpy as np
    from sklearn.model_selection import train_test_split

    @register_dataset("customized_qm9_dataset")
    def create_customized_dataset():
        dataset = QM9(label_keys=["mu"])
        num_graphs = len(dataset)

        # customized dataset split
        train_valid_idx, test_idx = train_test_split(
            np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
        )
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=num_graphs // 5, random_state=0
        )
        return {
            "dataset": dataset,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "source": "dgl"
        }

The function returns a dictionary. In the dictionary, ``dataset`` is the dataset object. ``train_idx`` is the graph indices used for training. Similarly we have
``valid_idx`` and ``test_idx``. Finally ``source`` records the underlying graph library used by the dataset. 

3. Specify the ``--user-data-dir`` as ``customized_dataset`` when training. And set ``--dataset-name`` as ``customized_qm9_dataset``.
Note that ``--user-data-dir`` should not be used together with ``--dataset-source``. All datasets defined in all python scripts under the ``customized_dataset``
will be registered automatically.
