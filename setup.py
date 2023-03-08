import os
import glob
import shutil
from setuptools import find_packages, Command

from numpy.distutils.core import Extension, setup


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

upper_files = [
    'upper.pyf',
    'Upper.f',
    'UpperLim.f',
    'C0.f',
    'CnMax.f',
    'CombConf.f',
    'y_vs_CLf.f',
    'CMaxinf.f',
    'ConfLev.f',
    'Cinf.f',
    'CERN_Stuff.f',
]

upper_f77_paths = []
for fname in upper_files:
    upper_f77_paths.append(f"darklim/limit/_upper/{fname}")

ext1 = Extension(
    name='darklim.limit._upper.upper',
    sources=upper_f77_paths,
)

upper_data_files = [
    'CMaxf.txt',
    'CMax.txt',
    'Cm.txt',
    'ymintable.txt',
    'y_vs_CLf.txt',
]

upper_data_paths = []
for fname in upper_data_files:
    upper_data_paths.append(f"darklim/limit/_upper/{fname}")

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        here = os.path.dirname(os.path.abspath(__file__))

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

setup(
    name="darklim",
    version="0.1.2",
    description="DM Limit Setting and Sensitivity",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Samuel Watkins",
    author_email="samwatkins@berkeley.edu",
    url="https://github.com/spice-herald/darklim",
    license_files = ('LICENSE', ),
    packages=find_packages(),
    zip_safe=False,
    cmdclass={
        'clean': CleanCommand,
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mendeleev',
        'scikit-learn',
    ],
    data_files=[
        ('darklim/limit/_upper/', upper_data_paths),
    ],
    ext_modules=[
        ext1,
    ],
)
