from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='long_gen',
      version='0.1.16',
      description='A library of functions for generating longitudinal data',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ],
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='generate data longitudinal time series synthetic',
      url='https://github.com/matthew-c-lenert/Long-Gen',
      author='MC Lenert',
      author_email='matthew.c.lenert@gmail.com',
      license='MIT',
      packages=['long_gen'],
      install_requires=[
          'numpy',
          'pandas',
          'sklearn',
          'scipy',
          'sklearn'
      ],
      zip_safe=False)

