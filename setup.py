from distutils.core import setup
import setuptools
from distutils.extension import Extension
import numpy

setup(
  name = 'OCAT',
  packages = ['OCAT'],
  version = '0.1.881',
  license='MIT',
  description = 'A new single-cell analytics framework',
  author = '',
  author_email = '',
  url = 'https://github.com/bowang-lab/OCAT',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/bowang-lab/OCAT/archive/v_0.1.28.tar.gz',    # I explain this later on
  keywords = ['RNA-SEQ', 'CLUSTERING', 'INTEGRATION'],
  install_requires=[
          'numpy>=1.17.2',
          'pandas',
          'scikit-learn>=0.21.3',
          'matplotlib',
          'scipy>=1.3.1',
          'cython',
          'umap-learn',
          'seaborn',
          'networkX'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
  ext_modules=[
        Extension(
            'OCAT.example',
            sources=['OCAT/example.pyx'],
        ),
    ],
  include_dirs = [numpy.get_include()]
)
