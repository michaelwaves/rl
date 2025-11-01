build for rocm
https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html

DOCKER_BUILDKIT=1 sudo docker build -f Dockerfile.rocm -t vllm-rocm .

Ok that didn't work, instead just use

```bash
docker pull rocm/vllm:latest
```

```sh
docker run -it --rm \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    rocm/vllm
    ```


docker run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
    rocm/vllm-dev:nightly
