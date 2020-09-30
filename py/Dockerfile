FROM pytorch/manylinux-cuda102

ENV BAZEL_VERSION=3.4.1
ENV JAVA_HOME=/etc/alternatives/java_sdk/bin/

RUN yum install -y \
    ninja-build \
    java-1.8.0-openjdk \
    java-1.8.0-openjdk-devel \
    zip

RUN wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo \
    && mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/

RUN mkdir /bazel \
    && cd /bazel \
    && wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-dist.zip -O bazel.zip \
    && unzip bazel.zip \
    && rm -f bazel.zip

ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.262.b10-0.el7_8.x86_64/ \
    PLATFORM="el6" \
    BAZEL_LINKLIBS="-l%:libstdc++.a"

RUN cd /bazel \
    && ./compile.sh


RUN mkdir /workspace

WORKDIR /workspace
ENV PATH=/bazel/output:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
