HyFile
======

API
---

.. autoclass:: hystorian.io.hyFile.HyFile
    :members:
    :special-members: __init__
    :exclude-members: Attributes
    :undoc-members:

.. autoclass:: hystorian.io.hyFile.HyFile.Attributes
    :members:
    :undoc-members:

How to extract data from file?
------------------------------

simply create a new HyFile (you can use the `with ... as ...` format, if the file does not exist, it will be created) then use `extract_data`, using the path to the file to convert as input

.. code-block:: python
    
    from pathlib import Path

    with HyFile('new_file.hdf5', 'r+') as f:
        f.extract_data(Path('/path/to/file/to/convert/1.ibw'))
        f.extract_data(Path('/path/to/file/to/convert/2.000'))
        f.extract_data(Path('/path/to/file/to/convert/3.ARDF'))

How to use apply and multiple_apply?
------------------------------------

Most of the time you should use `apply` in conjonction with `HyPath`, which are custom object which contains the path to some data inside an hdf5 (access through `HyPath.path`). 
If you use a string instead of an `HyPath` object, then the string itself will be passed to the function you want to compute.

For example, lets say that you want to sum all the elements of some dataset stored into your hdf5 file called `random.hdf5` in the path `datasets/data/grid`, you would do the following:

.. code-block:: python

    with HyFile('random.hdf5', 'r+') as f:
        f.apply(np.sum, HyPath('datasets/data/grid'), output_names = 'grid_sum')

`np.sum` can takes keyword arguments, like `axis`, you can simply pass them to `.apply`

.. code-block:: python

    with HyFile('random.hdf5', 'r+') as f:
        f.apply(np.sum, HyPath('datasets/data/grid'), output_names = 'grid_sum', axis=0)

Lets say that you have a second grid in your hdf5, and you want to some both of them together, you could do

.. code-block:: python

    with HyFile('random.hdf5', 'r+') as f:
        f.apply(np.sum, 
                [HyPath('datasets/data/grid'), HyPath('datasets/data/grid2')], 
                output_names = 'grid_sum')

Now lets say that instead you want to do the sum of the element of the first grid, then the element of the second grid, you could use `multiple_apply` (Note that now you will need to pass a list containing two output names)

.. code-block:: python

    with HyFile('random.hdf5', 'r+') as f:
        f.multiple_apply(np.sum, 
                        [HyPath('datasets/data/grid'), HyPath('datasets/data/grid2')],
                        output_names = ['grid_sum', 'grid_sum2'])
