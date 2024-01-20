FROM python:3.8

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update  && DEBIAN_FRONTEND=noninteractive apt-get -qqy install screen sox libopenblas-base cmake python3-pip mc wget libportaudio2 ffmpeg git less nano libsm6 libxext6 libxrender-dev  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app
COPY LightAutoML-0.3.7.4-py3-none-any.whl /app 

RUN pip install LightAutoML-0.3.7.4-py3-none-any.whl
RUN pip install llvmlite --ignore-installed
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["pwd"]
