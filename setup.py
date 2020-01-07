from distutils.util import convert_path

from setuptools import setup, find_packages


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
      ]
      )
