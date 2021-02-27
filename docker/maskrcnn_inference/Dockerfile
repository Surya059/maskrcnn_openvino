FROM mapler/caffe-py3:cpu

RUN python3 -m pip install scikit-build
RUN rmdir /workspace && mkdir /workspace && cd /workspace && git clone https://github.com/matterport/Mask_RCNN.git
RUN python3 -m pip install --upgrade pip
RUN cd /workspace/Mask_RCNN && python3 -m pip install -r requirements.txt && python3 setup.py install
RUN apt-get update
RUN apt-get update &&  \
    apt-get -o Dpkg::Options::="--force-confold" upgrade -q -y --force-yes && \
    apt-get -o Dpkg::Options::="--force-confold" dist-upgrade -q -y --force-yes

RUN python3 -m pip install numpy==1.16.4 Cython
RUN git clone https://github.com/waleedka/coco.git && cd coco/PythonAPI/ && python3 setup.py build_ext --inplace && python3 setup.py build_ext install 
RUN python3 -m pip install tensorflow==1.13.1
RUN python3 -m pip install keras==2.1.0
RUN apt-get install python3-tk
RUN rm -rf /workspace/coco
RUN mv /workspace /workspace_old
WORKDIR /workspace
