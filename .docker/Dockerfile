#FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
FROM skrjtech/python:debian-python3.9.16

SHELL [ "/bin/bash", "-c" ]

RUN apt update && apt -y upgrade 
RUN apt-get update && apt-get -y upgrade 
RUN apt install -y sudo git curl

CMD [ "/bin/bash" ]