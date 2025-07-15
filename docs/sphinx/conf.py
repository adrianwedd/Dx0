import os
import sys

project = 'Dx0'
author = 'MAI'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_typer',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'alabaster'

sys.path.insert(0, os.path.abspath('../..'))
