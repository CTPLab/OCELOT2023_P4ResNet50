# ResNet Ensemble Model - Cell Detector - OCELOT2023 Submission
This repository contains the python source code of the inference pipeline that we submitted to the test phase of the OCELOT2023 challenge.
An overview of the pipeline, its development and analysis in the context of the OCELOT2023 challenge is available in the OCELOT 2023 proceedings: [link TBD].
The pipeline relies on a single ensemble of three trained customized ResNet-50 models. The pipleine takes as input histology images of hematoxylin-and-eosin-stained tissue samples at resolution ~0.25um/px and outputs a list of the coordinates of the centers of the detected normal and tumor cells.


## References
\[1\] Lafarge, M.W and Koelzer, V.H., *Detecting Cells in Histopathology Images with a ResNet Ensemble Model.* [accepted manuscript] In: Proceedings of the OCELOT2023 Challenge (2023).

\[2\] *OCELOT 2023: Cell detection from cell-tissue interaction (2023)*, https://ocelot2023.grand-challenge.org/

\[3\] Ryu, J., Puche, A.V., Shin, J., Park, S., Brattoli, B., Lee, J., Jung, W., Cho, S.I., Paeng, K., Ock, C.Y., Yoo, D., Pereira, S.: *Ocelot: Overlapped cell on tissue dataset for histopathology.* In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)


## Content
- **run_example.py**: main script to run an example of the application of the cell detector.
- **model/**: package of the inference ensemble model (contains various customized modules and weight checkpoints).
- **example/000.tif**: example image used in the example script (can be copied from the public github repository of the OCELOT2023 Challenge: https://github.com/lunit-io/ocelot23algo).


## Requirements
This inference pipeline was integrated in a docker container, tested and submitted under the following configuration
- pytorch:1.8.1-cuda11.1-cudnn8
- numpy:1.22.3
- scikit-image:0.19.3