#!/usr/bin/env sh

docker run -it --rm --init --name nanomq -p 1883:1883 -p 8083:8083 -p 8883:8883 emqx/nanomq:latest-full
