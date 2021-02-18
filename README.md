# Multiscale

Please go see the wiki if you want to have more informations https://gitlab.unige.ch/paruch-group/multiscale/-/wikis/home

## Intoduction
Multiscale is a large scale project. Which has the motivation to gatherer all the snippets of code currently in the group under an organised library.

The project is based on the fact that, at the core our datas are only 1D, 2D, ND array, such that all the functions developped independently, could in theory be used for a lot of our datas.

The first part of the project is to create a file convert, which will convert most of the file type we use, into a well defined hdf5 format. Such that any data processing would be done using this format, for inter-compatibility.

## How to participate?
The easiest way to help develop this project would be to setup a github repository on your computer. On windows https://gitforwindows.org/ seems to work nicely.
Once installed, you need to create a ssh key pair to share your code with gitlab. https://gitlab.unige.ch/help/ssh/README#generating-a-new-ssh-key-pair

Once it is done, open Git Bash and clone the project by typing:

`git clone git@gitlab.unige.ch:paruch-group/multiscale.git /PATH/TO/THE/FOLDER/YOU/WANT/THE/CODE`

From there you can modify the code locally, and when necessary commit it to the server either by using Git GUI, or the usual git command lines.

## Uniformized structure of the files
The file contain:

* a **metadata** dataset, usually containing the unmodified metadata from the file

* a **type** dataset, containing a string with the extension name of the unconverted file

* a **data** folder containing **M dataset of dimension N**. Each of the dataset has a list of attributes. For the moment the attributes are the following:
	* name -- Name of the channel
	* shape -- Shape of the datas
	* size -- Physical size of the datas
	* offset -- Offset of the datas
	* unit -- Physical units of the datas
