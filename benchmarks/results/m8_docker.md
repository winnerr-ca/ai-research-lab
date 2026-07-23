# M8 Docker verification record: blocked by environment network policy

**Claim status: the CPU image build is NOT verified.** This record
exists so the limitation is documented with evidence instead of being
papered over.

## What was attempted (2026-07-23, commit `3d65e36` + results-in-progress tree)

The Docker daemon itself runs in the development environment
(`dockerd` 29.3.1, overlayfs). Three registries were attempted for the
base image; every one resolves its manifest but is denied at the
blob-CDN layer by the environment's egress gateway:

| Attempt | Result |
|---|---|
| `docker build -t rlcore:cpu .` (base `python:3.11-slim`, Docker Hub) | `production.cloudfront.docker.com: 403 Forbidden` (gateway CONNECT denial) |
| `docker pull public.ecr.aws/docker/library/python:3.11-slim` | `d2glxqk2uabbnd.cloudfront.net: Forbidden` |
| `docker pull ghcr.io/astral-sh/uv:python3.11-bookworm-slim` | `pkg-containers.githubusercontent.com: Forbidden` |

The proxy status endpoint records the same CONNECT rejections
(`connect_rejected`, "gateway answered 403"), confirming a network
policy denial rather than a Dockerfile defect.

## Consequence for claims

- `Dockerfile` (CPU) is **documented and syntactically standard but
  unbuilt here**; verify with `docker build -t rlcore:cpu . && docker
  run --rm rlcore:cpu --help` on any host with registry access.
- `Dockerfile.cuda` was already documented-unverified (no GPU); that is
  unchanged.
- No "verified container" claim appears anywhere in the documentation;
  `docs/DOCKER.md` states the same status as this record.
