from distutils.core import setup

setup(
    name='py2opencl',
    version='0.1.0',
    author='kieran hervold',
    author_email='hervold@gmail.com',
    packages=['py2opencl', 'py2opencl.test'],
    url='http://pypi.python.org/pypi/py2opencl/',
    license='LICENSE.txt',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "pyopencl >= 2014.1"
    ],
)
