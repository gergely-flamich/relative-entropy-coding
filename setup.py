from distutils.util import convert_path
from distutils.extension import Extension
from setuptools import setup, find_packages
import numpy

# https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
try:
    from Cython.Build import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("rec.io.entropy_coding", ["rec/io/entropy_coding.pyx"]),
    ]
    cmdclass.update({"build_ext": build_ext})

else:
    ext_modules += [
        Extension("rec.io.entropy_coding", ["rec/io/entropy_coding.c"]),
    ]


def readme() -> str:
    with open("README.md") as f:
        return f.read()


version_dict = {}
with open(convert_path("rec/version.py")) as f:
    exec(f.read(), version_dict)


setup(name="relative-entropy-coding",
      version=version_dict['__version__'],
      description="Relative Entropy coding implementations",
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=["Programming Language :: Python :: 3.6",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      author="Gergely Flamich",
      author_email="flamich.gergely@gmail.com",
      python_requires=">=3.6",
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "tensorflow-gpu",
          "tensorflow-probability",
      ],
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      include_dirs=[numpy.get_include()],
      )
