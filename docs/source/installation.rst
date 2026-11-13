Installation
============

Prerequisites
-------------

Before installing ``zagar``, ensure you have:

- **Python** >= 3.8
- **C++17 compatible compiler** (GCC >= 7, Clang >= 5, or MSVC >= 2017)
- **CMake** >= 3.15
- **Boost** (header-only, for multiprecision arithmetic)

Installing Boost
~~~~~~~~~~~~~~~~

**macOS** (using Homebrew):

.. code-block:: bash

   brew install boost

**Ubuntu/Debian**:

.. code-block:: bash

   sudo apt-get install libboost-dev

**Fedora/RHEL**:

.. code-block:: bash

   sudo dnf install boost-devel

**Windows** (using vcpkg):

.. code-block:: bash

   vcpkg install boost

**Conda** (any platform):

.. code-block:: bash

   conda install -c conda-forge boost

Optional: OpenMP for Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For parallel computation support via ``getA12_parallel``, install OpenMP:

**macOS**:

.. code-block:: bash

   brew install libomp

**Ubuntu/Debian**:

.. code-block:: bash

   sudo apt-get install libomp-dev

**Fedora/RHEL**:

.. code-block:: bash

   sudo dnf install libomp-devel


Python Users
------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install ``zagar`` from PyPI with:

.. code-block:: bash

   pip install zagar

.. note::

   If Boost is not installed, you will see an error like::

      CMake Error at CMakeLists.txt:52 (message):
        Boost not found.  Please install Boost:

          macOS:   brew install boost
          Ubuntu:  sudo apt-get install libboost-dev
          conda:   conda install -c conda-forge boost

   Install Boost first (see Prerequisites above), then retry ``pip install zagar``.

Example: Installing in a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a complete example of installing ``zagar`` in a fresh conda environment:

.. code-block:: bash

   # Create and activate a new environment
   conda create -n zagar-test python=3.11 -y
   conda activate zagar-test

   # Install Boost (required dependency)
   conda install -c conda-forge boost

   # Install zagar from PyPI
   pip install zagar

   # Verify installation
   python -c "import zagar; print(zagar.__version__)"

From Source
~~~~~~~~~~~

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/alanaw1/zagar.git
   cd zagar
   pip install .

For development (editable install with test dependencies):

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

After installation, verify that ``zagar`` works correctly:

.. code-block:: python

   import zagar

   # Check version
   print(zagar.__version__)

   # Test basic functionality
   result = zagar.count_binary_matrices([2, 1], [1, 1, 1])
   print(result)  # {'number': 3, 'nvec': [2, 1], 'mvec': [1, 1, 1]}

   # Check if OpenMP is available
   print(f"OpenMP support: {zagar.has_openmp()}")


SageMath Users
--------------

`SageMath <https://doc.sagemath.org/html/en/index.html>`_ includes its own Python environment, so ``zagar`` must be installed
into Sage's Python rather than your system Python.

Method 1: Using Sage's pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, ensure Boost is installed on your system (see Prerequisites above).
Then install ``zagar`` using Sage's built-in pip:

.. code-block:: bash

   # Install from PyPI (once published)
   sage -pip install zagar

   # Or install from source
   git clone https://github.com/alanaw1/zagar.git
   cd zagar
   sage -pip install .

Method 2: Using sage.repl.ipython_kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using SageMath in a Jupyter notebook, run a shell command in a cell:

.. code-block:: text

   # In a Sage notebook cell, use shell escape
   !sage -pip install zagar

Verifying Installation in SageMath
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start a Sage session and test the installation:

.. code-block:: bash

   sage

Then in the Sage prompt:

.. code-block:: python

   sage: import zagar
   sage: zagar.__version__
   '0.2.0'
   sage: zagar.count_binary_matrices([2, 1], [1, 1, 1])
   {'number': 3, 'nvec': [2, 1], 'mvec': [1, 1, 1]}

Using zagar with Sage Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``zagar`` functions accept standard Python lists. When working with Sage
integers or vectors, convert them to Python lists first:

.. code-block:: python

   sage: import zagar
   sage:
   sage: # Sage integers work directly in lists
   sage: N = 5
   sage: a_vec = [2, 2]
   sage: phi0_vec = [1, 1]
   sage: phi1_vec = [1, 1]
   sage:
   sage: result = zagar.getA12(N, a_vec, phi0_vec, phi1_vec)
   sage: print(result)
   7632
   sage:
   sage: # For Sage vectors, convert to list
   sage: v = vector([1, 2, 3])
   sage: python_list = list(v)

The return values from ``zagar`` are standard Python integers with arbitrary
precision (using Boost.Multiprecision internally), which integrate seamlessly
with Sage's integer arithmetic.


R Users
-------

Coming soon!


Troubleshooting
---------------

Boost Not Found
~~~~~~~~~~~~~~~

If you see an error like "Boost not found", ensure Boost is installed and
its headers are in a standard location. You can also set the path manually:

.. code-block:: bash

   # Set Boost include path before installing
   export Boost_INCLUDE_DIRS=/path/to/boost/include
   pip install .

OpenMP Not Available
~~~~~~~~~~~~~~~~~~~~

If ``zagar.has_openmp()`` returns ``False`` but you have OpenMP installed:

**macOS**: Ensure libomp is installed via Homebrew and rebuild:

.. code-block:: bash

   brew install libomp
   pip uninstall zagar
   pip install . --no-cache-dir

**Linux**: Install the OpenMP development package for your distribution.

Compilation Errors
~~~~~~~~~~~~~~~~~~

If you encounter C++ compilation errors:

1. Ensure you have a C++17 compatible compiler
2. On macOS, ensure Xcode Command Line Tools are installed:

   .. code-block:: bash

      xcode-select --install

3. On Windows, ensure Visual Studio Build Tools 2017 or later are installed
