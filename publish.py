#!/usr/bin/env python

# Python analogue of MATLAB publish
# Copyright 2023 Lawrence Mitchell <wence@gmx.li>, 2023- Patrick Farrell <patrick.farrell@maths.ox.ac.uk>
# Released under BSD 3-clause license
# 2024-09-23

import inspect
import jupytext
import subprocess
import sys
import pathlib

# Dependencies:
# pip install ipykernel nbconvert jupytext


def nbconvert(script):
    """
    Call jupytext to convert the .py file to an IPython notebook,
    then use nbconvert to execute and save the IPython notebook to HTML.
    """

    with open(script, "r", encoding="utf-8") as f:
        lines = f.readlines()

    script_data = jupytext.reads("".join(lines), fmt="py:light")

    nb = script.with_suffix(".ipynb")
    jupytext.write(script_data, nb, fmt="ipynb")
    subprocess.check_call([sys.executable, "-m", "nbconvert", "--execute", "--no-prompt", "--to", "html", str(nb)])


def render(obj, name=None):
    """
    Render a sympy object in the published document.

    This works by getting sympy to give a latex representation,
    and then using IPython's display features to make a math cell
    for it.
    """

    from IPython.display import display, Math
    import sympy as sp

    if name:
        prestring = name + " = "
    else:
        prestring = ""

    display(Math("$$" + prestring + sp.latex(obj) + "$$"))


if __name__ == "__main__":
    import sys
    script = pathlib.Path(sys.argv[1])
    nbconvert(script)
