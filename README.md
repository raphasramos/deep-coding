PREPARING A VIRTUALENV TO THE PROJECT IN LINUX / MACOS

This tutorial considers you've installed virtualenv in the system with:
pip3 install virtualenv 

It's interesting to create a virtualenv for the project. Enter the root of the
repository and Use the following command:
virtualenv venv --system-site-packages

It`ll create a folder named venv inside the root of the repository. The .gitignore
already ignorar a venv folder. So it's good to create it this way.

After that, use the following command to use the env:
source venv/bin/activate

With this, you`ll be using the venv libs. You can use the following command to
install the packages for the project:
pip3 install -r requirements.txt

In windows, the installation of virtualenv, the creation of the env and the
activation of the env are different. But the idea is the same.
