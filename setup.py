from distutils.core import setup

setup(name='Beam search',
      version='0.1.0',
      description='Beam search for neural network sequence to sequence (encoder-decoder) models.',
      long_description=open("README.md").read(),
      author='Ottokar Tilk',
      author_email='ottokart@gmail.com',
      license="MIT",
      py_modules=['beam_search'],
      install_requires=[
        "numpy",
      ],
     )