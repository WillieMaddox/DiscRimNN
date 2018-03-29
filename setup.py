import sys
from setuptools import setup
from setuptools import find_packages
# from setuptools.command.test import test as TestCommand

NAME = 'MixSig'
VERSION = '0.1.0'
DESCRIPTION = 'Explore RNN\'s using mixed waves.'
LICENSE = 'MIT'
URL = 'https://github.com/WillieMaddox/MixSig'
AUTHOR = 'Willie Maddox'
EMAIL = 'willie.maddox@gmail.com'
KEYWORDS = ''


# class PyTest(TestCommand):
#     def initialize_options(self):
#         TestCommand.initialize_options(self)
#         self.pytest_args = []
#
#     def run_tests(self):
#         sys.exit(0)


def setup_package():
    # cmdclass = {'test': PyTest}

    setup(
        name=NAME,
        url=URL,
        version=VERSION,
        license=LICENSE,
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description='',
        setup_requires=['pytest-runner'],
        install_requires=['numpy', 'pandas', 'matplotlib', 'jupyter'],
        classifiers=[],
        packages=[package for package in find_packages() if package.startswith('mixsig')],
        tests_require=['pytest', 'hypothesis'],
        test_suite='',
        # cmdclass=cmdclass,
        command_options={},
        entry_points={},
        zip_safe=False
    )


if __name__ == '__main__':
    setup_package()
