import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pix_mclass",
    version="0.0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="Multiclass pixel classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/talissman",
    download_url = 'https://github.com/jeanollion/pix_mclass/archive/0.0.1.tar.gz',
    packages=setuptools.find_packages(),
    keywords = ['Segmentation', 'Classification', 'Microscopy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras_preprocessing', 'dataset_iterator>=0.2.2', 'elasticdeform']
)
