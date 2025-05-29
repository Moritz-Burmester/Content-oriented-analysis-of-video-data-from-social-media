This repository is in connection with my bachelor thesis.
The goal is to test the capabilities of the models Video-LLaVA, Video-ChatGPT, PandaGPT, and CLIP in classifying videos.

The code was used in the following order:

1.) Finding Duplicates in the Dataset
a) video_hashes.py is used to calculate the hash values of the first and last non-black frame of each video.
b) duplicates.py is used to group duplicates with their original video and to create a file with all duplicates and their IDs.

The results and code can be found in the repository under /Duplicates_and_HashValues/.

2.) Classifying the Dataset
a) main.py is used for the main classification task. First, the corresponding conda environment of the model needs to be activated. Then the model is initialized and the classification begins. The returned outputs are then formatted and saved in solution files.
b) classification_modelname.py is used for initializing the model and classifying the video. The output is returned to main.py.

Depending on the use case, some alterations need to be made to the main.py file. What can be changed is mentioned in the file.

3.) Working with the classification data
a) analysis.py is used for generating all kinds of information about the results. The file includes methods to visualize and print information used in the thesis.
