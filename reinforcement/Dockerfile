FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c", "bazel run //:reinforcement \
  -- --base_dir=$(pwd)/results/$(date +%Y-%m-%d)"]
