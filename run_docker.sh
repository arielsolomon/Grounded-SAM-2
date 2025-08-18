docker run --gpus all -it --rm --net=host --privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v "/Data/Projects/Maavarim":/work \
-w "/work/" \
-e DISPLAY=ISPLAY \
--name=gsa \
--ipc=host -it grounded_sam2:1.0
