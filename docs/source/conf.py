# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os 
import sys
sys.path.insert(0, os.path.abspath('../../pyoomph'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyoomph'
copyright = '2024, Christian Diddens & Duarte Rocha'
author = 'Christian Diddens & Duarte Rocha'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    "sphinx.ext.autosectionlabel",
    'sphinxcontrib.bibtex',
    # 'sphinx.ext.imgmath',
    ]

if False:
	extensions+=[    "sphinx_codeautolink"]
	codeautolink_global_preface="from pyoomph import *"
	codeautolink_autodoc_inject=True
	
# todo_include_todos = True

# imgmath_image_format = 'svg'

bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
modindex_common_prefix = ['pyoomph.']

# Do not inherit e.g. the define_residuals or something for InitialCondition, etc
autodoc_inherit_docstrings=False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

numfig = True

exclude_patterns=["latex_tutorial.rst"]

latex_engine = 'xelatex'

latex_documents = [ ('latex_tutorial', 'pyoomph_tutorial.tex', 'Pyoomph Tutorial', 'Christian Diddens and Duarte Rocha', 'manual', True)]
latex_domain_indices = False
# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = './_static/banner-large.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False
latex_toplevel_sectioning = 'chapter'




#from dataclasses import dataclass, field
#import sphinxcontrib.bibtex.plugin
#from sphinxcontrib.bibtex.style.referencing import BracketStyle
#from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle
#def bracket_style() -> BracketStyle:
#    return BracketStyle(
#        left='(',
#        right=')',
#    )


#@dataclass
#class MyReferenceStyle(AuthorYearReferenceStyle):
#    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
#    bracket_textual: BracketStyle = field(default_factory=bracket_style)
#    bracket_author: BracketStyle = field(default_factory=bracket_style)
#    bracket_label: BracketStyle = field(default_factory=bracket_style)
#    bracket_year: BracketStyle = field(default_factory=bracket_style)


#sphinxcontrib.bibtex.plugin.register_plugin('sphinxcontrib.bibtex.style.referencing','author_year_round', MyReferenceStyle)
#bibtex_reference_style = 'author_year_round'


def set_master_doc(app):
    if app.tags.has("latex"):
        app.config.master_doc = "latex_tutorial"
        app.config.exclude_patterns.remove("latex_tutorial.rst")
        app.config.exclude_patterns.append("tutorial.rst")
        app.config.exclude_patterns.append("index.rst")

def setup(app):
    app.connect('builder-inited', set_master_doc)
