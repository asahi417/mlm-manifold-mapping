from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

VERSION = '0.0.0'
setup(
    name='m3',
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    license='MIT',
    description="Iterative text revision based on the mapping onto manifold of masked language model's low-perplexity space.",
    url='https://github.com/asahi417/mlm-manifold-mapping',
    download_url="https://github.com/asahi417/bertprompt/archive/v{}.tar.gz".format(VERSION),
    keywords=['language-model', 'nlp', 'bert', 'text-generation'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'sentencepiece',
        "transformers",
        "torch",
        "tqdm",
        "pandas",
        "requests",
        "datasets",
        "sklearn",
        "wandb",
        "ray[tune]",
        "textattack"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'm3-rewrite = m3.m3_cl.rewrite_text:main',
            'm3-rewrite-basic-augmenter = m3.m3_cl.rewrite_text_basic_augmenter.py:main'
        ]
    }
)
