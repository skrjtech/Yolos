FROM ubuntu:latest

SHELL [ "/bin/bash", "-c" ]

RUN apt update && apt upgrade -y
RUN apt-get update && apt-get upgrade -y
RUN apt install -y sudo git curl locales vim tmux

RUN bash <(curl -sSL https://raw.githubusercontent.com/skrjtech/Archives/main/Installer/python.sh) 3.9.18
RUN bash <(curl -sSL https://raw.githubusercontent.com/skrjtech/Archives/main/Installer/pycharm.sh) 2023.2.1

RUN python -m venv /aios && source /aios/bin/activate && pip install --upgrade pip
COPY ../../requirements.txt /aios/requirements.txt
RUN /aios/bin/pip install -r /aios/requirements.txt

CMD [ "/bin/bash" ]