FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3
WORKDIR /reinforcement

COPY . .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c", "bazel run //:reinforcement -- --base_dir=/results/$(date +%Y-%m-%d)"]
