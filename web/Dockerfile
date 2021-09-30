# base image
# a little overkill but need it to install dot cli for dtreeviz
FROM python:3.6-slim  

#set the timezone
ENV TZ=Europe/Berlin

# exposing default port for streamlit
EXPOSE 8501

# copying all files over
RUN mkdir web
COPY . /web
WORKDIR /web

# install pip then packages
RUN pip install --upgrade pip --no-cache-dir \
 && pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir \
 && pip install numpy pandas streamlit scikit-learn --no-cache-dir \
 && cp /usr/share/zoneinfo/${TZ} /etc/localtime \
 && echo "${TZ}" > /etc/timezone
      
#RUN conda config --add channels conda-forge && \
##    conda config --set channel_priority true && \
#    conda install --quiet --yes pytorch torchvision cpuonly -c pytorch && \
#     conda install --quiet --yes -c conda-forge/label/dev numpy  && \
#    conda install --quiet --yes -c conda-forge streamlit    && \
#    conda install --quiet --yes pandas    && \
#    conda install --quiet --yes -c conda-forge scikit-learn  &&\
#    conda update --quiet --yes --all \
#    && \
#   conda clean --all -f -y


# cmd to launch app when container is run
# CMD ["streamlit", "run", "app.py","--server.port","8000"] 
ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]
