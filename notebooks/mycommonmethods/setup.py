from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Python package for my diploma thesis'
LONG_DESCRIPTION = 'Python package for my diploma thesis that contains some common methods that I use in different python notebooks'

# Setting up
setup(
        name="commonmethods", 
        version=VERSION,
        author="Fanni Molnár",
        author_email="<fannimolnr98@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['scikit-learn', 'numpy', 'opencv-python']
)