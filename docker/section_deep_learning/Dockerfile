FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN conda install pyg -c pyg -y && conda clean -ya
RUN conda install nodejs "h5py<3.2" jupyterlab ipywidgets tensorboard -c conda-forge -y && conda clean -ya

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

ENTRYPOINT jupyter notebook --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.notebook_dir="/workspace" --ip 0.0.0.0 --allow-root --no-browser --port 8888 & /bin/bash
