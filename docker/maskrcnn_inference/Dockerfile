FROM mapler/caffe-py3:cpu

RUN python3 -m pip install scikit-build
RUN rmdir /workspace && mkdir /workspace && cd /workspace && git clone https://github.com/matterport/Mask_RCNN.git
RUN python3 -m pip install --upgrade pip
RUN cd /workspace/Mask_RCNN && python3 -m pip install -r requirements.txt && python3 setup.py install
RUN mv /workspace /workspace_old
WORKDIR /workspace
