import setuptools
from glob import glob

# Will load the README.md file into a long_description of the package
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# Load the requirements file
with open('requirements.txt') as f:
    required = f.read().splitlines()
if __name__ == "__main__":
    setuptools.setup(
        name='NNAIMGUI',
        version='0.0.1',
        author='Miguel Gallegos',
        author_email='gallegosmiguel@uniovi.es',
        description="A GUI interfaced code for the prediction, equilibration and visualization of QTAIM properties with Neural Networks",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/m-gallegos/NNAIMGUI',
        project_urls = {
            "NNAIMQ": "https://github.com/m-gallegos/NNAIMQ"
        },
        license='MIT',
        install_requires=required,
        zip_safe= False,
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        package_data={'NNAIMGUI': [
                'data/logo.png',
                'models/NNAIMQ/nn*',
                'models/NNAIMQ/input.*',
                'examples/LIAIM/input.*',
                'examples/LIAIM/nn*',
                'examples/charge_equilibration/*.py',
        ]}
    )
