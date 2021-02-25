FROM mapler/caffe-py3:cpu

RUN python3 -m pip install scikit-build
RUN mkdir /workspace && cd /workspace && git clone https://github.com/matterport/Mask_RCNN.git
RUN cd /workspace/Mask_RCNN && python3 -m pip install requirements.txt && python3 setup.py install
WORKDIR /workspace
