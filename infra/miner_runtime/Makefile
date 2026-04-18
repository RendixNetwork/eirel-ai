IMAGE ?= registry.eirel.internal/miner-runtime:v1
SDK_ROOT ?= ../../../eirel

.PHONY: build-runtime-image push-runtime-image

build-runtime-image:
	docker build -t $(IMAGE) -f $(SDK_ROOT)/Dockerfile.runtime $(SDK_ROOT)

push-runtime-image: build-runtime-image
	docker push $(IMAGE)
