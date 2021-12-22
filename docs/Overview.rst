Overview
========

Basically, Graphormer inherits the extending usage of fairseq, which means it could easily support user-defined `plug-ins <https://fairseq.readthedocs.io/en/latest/overview.html>`_.  

For example, the Graphormer-base model could be defined through :class:`~graphormer.models.GraphormerModel`, which inherits the :class:`~fairseq.models.FairseqModel` class.


It's also easy to extend the Graphormer-base model, which means you could define your own `model <https://fairseq.readthedocs.io/en/latest/models.html>`_ and `criterion <https://fairseq.readthedocs.io/en/latest/criterions.html>`_, and then use them in Graphormer.

Also, development of new model is easy. We provide a tutorial of how to implement a simple MLP model on graph in :ref:`Tutorials`.
