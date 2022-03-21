# Docker

Original source: [Det3D](https://github.com/TRAILab/Det3D)

## Setup
To use PDV in docker please make sure you have `nvidia-docker` installed which can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). In a nutshell:

1. Install Docker
```
curl https://get.docker.com | sh \
  && sudo systemctl start docker \
  && sudo systemctl enable docker
```

2. Setup the `stable` repository and the GPG key:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

3. Install `nvidia-docker2`:
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

4. Restart Docker daemon to complete installation after setting the default runtime:
```
sudo systemctl restart docker
```

5. Ensure `nvidia-docker` is working with:
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Build Docker Image
```
cd PDV/docker/
bash build.sh
```

## Run Docker Container
```
cd PDV/docker/
bash run.sh
```
