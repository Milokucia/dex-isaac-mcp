# Isaac Lab base image; the repo itself is bind-mounted by docker-compose.yaml.
#
# Pull requires an NGC login:  docker login nvcr.io  (user: $oauthtoken, pass: <NGC API key>)
#
# Pinned by DIGEST, not just tag: a tag is mutable. A rebuild of `2.3.2` months
# later once pulled an image whose /isaac-sim was mode 750 owned by uid 1234,
# and every non-root `docker compose run` failed with
#   [ERROR] Unable to find any Python executable at path: '/workspace/isaaclab/_isaac_sim/python.sh'
# Bump the digest as a deliberate upgrade.
FROM nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2

# Make /isaac-sim traversable by the unprivileged user compose runs as.
# Exactly one directory needs it; everything beneath is already world-readable.
USER root
RUN chmod o+rx /isaac-sim

WORKDIR /workspace/dex-isaac-mcp
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh", "-p"]
CMD ["scripts/simd.py", "--headless"]
