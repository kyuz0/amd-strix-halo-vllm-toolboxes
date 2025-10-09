.PHONY: rebuild push pull all clean

DOCKER_USER := your-docker-username
IMAGE_NAME := vllm-therock-gfx1151-aotriton
DOCKERFILE := Dockerfile.vllm-therock-gfx1151-aotriton
TEMP_IMAGE := temp
FULL_IMAGE := $(DOCKER_USER)/$(IMAGE_NAME)

all: clean rebuild push pull

clean:
	@echo "Removing existing vllm container..."
	-podman rm -f vllm
	-toolbox rm vllm

rebuild:
	@echo "Building Docker image..."
	docker build -f $(DOCKERFILE) -t $(TEMP_IMAGE) . --no-cache

tag: rebuild
	@echo "Tagging image..."
	docker tag $(TEMP_IMAGE) $(FULL_IMAGE):latest

push: tag
	@echo "Pushing to Docker Hub..."
	docker push $(FULL_IMAGE):latest

pull: push
	@echo "Pulling image into Podman..."
	podman pull docker.io/$(FULL_IMAGE)

create: pull
	@echo "Creating toolbox..."
	toolbox create vllm --image docker.io/$(FULL_IMAGE) -- --device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined
