import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PixMClass",
    version="0.1.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="Multiclass pixel classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/pix_mclass",
    download_url='https://github.com/jeanollion/pix_mclass/releases/download/v0.1.1/pix_mclass-0.1.1.tar.gz',
    packages=setuptools.find_packages(),
    keywords = ['Segmentation', 'Classification', 'Microscopy', 'Cell'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow>=2.7.1', 'dataset_iterator>=0.4.0', 'elasticdeform']
)
