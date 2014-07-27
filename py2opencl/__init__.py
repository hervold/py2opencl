
# see http://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package
from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution('foobar')
    if not __file__.startswith(os.path.join(_dist.location, 'foobar')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version

from .driver import Py2OpenCL
