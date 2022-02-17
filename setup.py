from setuptools import setup, find_packages


requirements = [
    'setuptools',
    'tqdm',
    'click',
    'scipy',
    'numpy',
    'statsmodels'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest']

setup(
    author="M. Hasan Çelik",
    author_email='muhammedhasancelik@gmail.com',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Beta Binomial test to detect aberration in count data",
    install_requires=requirements,
    license="MIT license",
    keywords=['beta-binomial', 'counts', 'statistics'],
    name='betabinomial',
    packages=find_packages(include=['betabinomial']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/muhammedhasan/betabinomial',
    version='0.0.2',
)
