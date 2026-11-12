# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# The zagar package must be installed for autodoc to work, because the _core
# C++ extension module only exists after building/installing. Run:
#   pip install -e .  (for development install)
# or:
#   pip install .     (for regular install)
# before building the documentation.


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'zagar'
copyright = '2026, Alan J. Aw'
author = 'Alan J. Aw'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if using Google/NumPy docstrings
    'sphinx.ext.viewcode',
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo_dark.png",
}
html_favicon = "_static/favicon.ico"