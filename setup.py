try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from setuptools import find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name='wikirec',
    version='0.0.1',
    description='Open-source recommendation engines based on Wikipedia data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(),
    license='new BSD',
    url="https://github.com/andrewtavis/wikirec",
    author='Andrew Tavis McAllister',
    author_email='andrew.t.mcallister@gmail.com'
)

install_requires = [
    
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)