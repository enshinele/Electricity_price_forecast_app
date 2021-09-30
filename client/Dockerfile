# base image
# a little overkill but need it to install dot cli for dtreeviz
FROM python:3.6-slim  

# exposing default port for client
EXPOSE 8080

# copying all files over
RUN mkdir client
COPY . /client
WORKDIR /client

# install pip then packages
RUN pip install --upgrade pip --no-cache-dir && pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir && pip install numpy pandas scikit-learn --no-cache-dir && pip install requests --no-cache-dir
      
CMD ["python","client.py"]
