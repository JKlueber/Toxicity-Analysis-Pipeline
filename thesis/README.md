Code Thesis Notes
=================

Date: October 25th, 2022


Science Craftsmanship
---------------------
  - [How to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
  - [Things a scientist is suppposed to know](https://github.com/philippbayer/Things-a-scientist-is-suppposed-to-know) or material.
  - [Resources to Help Global Equality for PhDs in NLP / AI](https://github.com/zhijing-jin/nlp-phd-global-equality)
  - [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)
  - [Jimmy Lin's Guide to Science](https://github.com/lintool/guide)
  - [Warning Signs in Experimental Design](https://norvig.com/experiment-design.html)

Technical Writing Guides
------------------------
  - [Google's Technical Wrting Courses for Engineers](https://developers.google.com/tech-writing)
  - [Writing a Thesis with LaTex](http://www.tug.org/pracjourn/2008-1/mori/) or material.
  - [Writing for Computer Science](https://archive.org/details/springer_10.1007-978-0-85729-422-7/mode/2up) or material.

Organizing Literature
------------------------
See our [guide](https://git.webis.de/code-generic/code-webis-faq#how-to-do-literature-research) on literature research.


Organizing Code and data
------------------------
  - [Cookiecutter Project Template](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)

Style Guides
------------
  - [The Chicago Manual of Style](http://www.chicagomanualofstyle.org)
  - [Wikipedia, Book Design](http://en.wikipedia.org/wiki/Book_design)
  - [A detailed guide to colors in data vis style guides](https://blog.datawrapper.de/colors-for-data-vis-style-guides/)


Plotting
--------
  - [Gnuplot](http://gnuplot.info/)
  - [Matplotlib](http://matplotlib.sourceforge.net/)
  - [R](http://www.r-project.org/)


Compilation
-----------
  1. Commands:
  ```
    pdflatex thesis
    bibtex thesis
    pdflatex thesis
    pdflatex thesis
    pdflatex thesis
  ```
  2. This should create the file thesis.pdf.
  3. If not, make sure you have all the require packages installed (see webisthesis.cls).
  4. If everything works like it should, you should not be forced to change anything in webisthesis.cls at all.


Overleaf
--------

This template can also be found in the Overleaf Gallery. If you use Overleaf, make sure to share a link to your thesis in your Gitlab repository's readme so that your supervisor(s) can find it quickly. Also make sure to copy the LaTex source code from Overleaf into your Gitlab repository when you are done.
- [Overleaf Template](https://www.overleaf.com/latex/templates/webis-thesis-template/nfhzrtmtzjty)
