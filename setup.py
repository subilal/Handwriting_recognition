from setuptools import setup, find_packages

setup(
    name="HandwritingRecognition",
    version="1.0",
    packages=find_packages(),


    entry_points={
        'console_scripts': [
            'preprocessing = Preprocess:preprocess',
        ]
    }
)
