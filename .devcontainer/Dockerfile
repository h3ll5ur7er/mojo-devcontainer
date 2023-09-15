# devcontainer ubuntu python
FROM ubuntu:22.04

ARG USERNAME=mojo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV MODULAR_AUTH
ENV MODULAR_HOME="/home/mojo/.modular"
ENV MOJO_HOME="${MODULAR_HOME}/pkg/packages.modular.com_mojo"
ENV PATH="${MOJO_HOME}/bin:$PATH"
ENV MOJO_PYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.10.so"

RUN apt update && apt install -y python3-dev python3-pip build-essential git curl wget apt-transport-https
RUN curl -1sLf 'https://dl.modular.com/bBNWiLZX5igwHXeu/installer/gpg.0E4925737A3895AD.key' |  gpg --dearmor >> /usr/share/keyrings/modular-installer-archive-keyring.gpg
RUN curl -1sLf 'https://dl.modular.com/bBNWiLZX5igwHXeu/installer/config.deb.txt?distro=debian&codename=wheezy' > /etc/apt/sources.list.d/modular-installer.list
RUN apt update && apt install -y modular

RUN python3 -m pip install numpy matplotlib pandas seaborn
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /home/mojo

RUN modular auth ${MODULAR_AUTH}

RUN modular install mojo

# ENTRYPOINT [ "python3" ]
# ENTRYPOINT [ "/bin/bash" ]
ENTRYPOINT [ "mojo" ]
