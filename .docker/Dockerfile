FROM ubuntu:latest

SHELL [ "/bin/bash", "-c" ]

RUN apt update && apt upgrade -y
RUN apt-get update && apt-get upgrade -y
RUN apt install -y sudo git curl locales vim tmux language-pack-ja-base language-pack-ja fonts-noto-cjk fcitx-mozc

# Tab補完の使用
RUN apt install -y bash-completion && source /etc/bash_completion  && echo "source /etc/bash_completion" >> ~/.bashrc

RUN echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc
RUN echo 'export GTK_IM_MODULE=xim' >>> ~/.bashrc
RUN echo 'export QT_IM_MODULE=fcitx' >>> ~/.bashrc
RUN echo 'export XMODIFIERS="@im=fcitx"' >>> ~/.bashrc
RUN echo 'export DefalutIMModule=fcit' >>> ~/.bashrc

RUN bash <(curl -sSL https://raw.githubusercontent.com/skrjtech/Archives/main/Installer/python.sh) 3.9.18
# RUN bash <(curl -sSL https://raw.githubusercontent.com/skrjtech/Archives/main/Installer/pycharm.sh) 2023.2.1

RUN /usr/bin/python -m venv /aios 
RUN update-alternatives --install /aios/bin/python python /usr/bin/python 1
# RUN echo "source /aios/bin/activate" >> ~/.bashrc
RUN source ~/.bashrc && /aios/bin/pip install --upgrade pip
COPY ../../requirements.txt /aios/requirements.txt
RUN /aios/bin/pip install -r /aios/requirements.txt

RUN mkdir /yolos
WORKDIR /yolos

CMD [ "/bin/bash" ]