# AMD Strix Halo InfiniBand Cluster Setup Guide

This guide covers running the two-node **Strix Halo** vLLM cluster over a **native
InfiniBand** fabric, as an alternative to the [Ethernet / RDMA-RoCE guide](setup_guide.md).

The toolbox container already ships everything needed on the data path
(`rdma-core`, the libibverbs providers, `infiniband-diags`, `perftest`), and
`refresh_toolbox.sh` exposes `/dev/infiniband` into the container when it detects
an IB link, so no container changes are required. This guide covers the host-side
fabric and how to select the **InfiniBand** transport in the cluster launcher.

## When to use this guide

Use native InfiniBand when you have a dedicated IB **HCA** (Host Channel
Adapter — the network card that speaks InfiniBand) on both nodes. Compared to
RoCE over a converged Ethernet NIC, native IB keeps the RDMA data path off the
routed Ethernet network entirely.

> Reference implementation: RCCL runs the TP all-reduce over the fabric with a
> native-IB HCA on both nodes. The transport wiring below is the validated
> configuration for this cluster setup.
>
> Refer to the the community wiki for details of a low budget infiniband hardware setup:
> <https://strixhalo.wiki/AI/Clustering_with_RDMA>

## Table of Contents

1. [Fabric requirements](#1-fabric-requirements)
2. [Host configuration](#2-host-configuration)
3. [Verify the fabric from the host](#3-verify-the-fabric-from-the-host)
4. [Selecting the InfiniBand transport](#4-selecting-the-infiniband-transport)
5. [What the launcher sets (and why)](#5-what-the-launcher-sets-and-why)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Fabric requirements

*   **Nodes**: 2x Strix Halo hosts (see the [main setup guide](setup_guide.md) for
    the base host/kernel/BIOS configuration).
*   **HCAs**: one InfiniBand HCA per node. Both ports must be up and cabled
    together, or plugged into an IB switch.
*   **Subnet Manager (opensm)**: native IB has **no self-managed fabric** — one
    node (typically the head, here `fuzzy`) must run an InfiniBand Subnet Manager
    (`opensm`). Without it the ports will not reach `ACTIVE` state.
*   **Control plane**: Ray / GLOO rendezvous and the management interface still
    run over Ethernet (the interface carrying `head_ip`/`worker_ip`). Only the
    RCCL data plane runs over the IB fabric.

## 2. Host configuration

These steps run on the **host OS** (not inside the toolbox) of both nodes.

### 2.1 Install userspace tools

```bash
sudo dnf install rdma-core libibverbs-utils perftest infiniband-diags
```

> The container image already carries these (plus the validated rdma-core v62
> overlay), but the host needs them for `ibv_devinfo`, `ibstat`, and `opensm`.

### 2.2 Run a Subnet Manager on one node

On the **head** node (one node only), enable and start `opensm`:

```bash
sudo systemctl enable --now opensm
sudo systemctl status opensm
```

On a single-cable direct-connect setup this brings both ports to `ACTIVE`. Do
**not** run `opensm` on both nodes unless you configure a master/slave policy.

### 2.3 Bring the IB link up (no IP required)

Native IB needs no IP address for RCCL to use it. The ports just need to be
`ACTIVE`. If the kernel did not enable the ports automatically:

```bash
sudo ibportstate -D 0 1 enable     # or via udev/ibft, see below
```

### 2.4 (Optional) Persistent HCA naming

If you have several RDMA devices (e.g. USB4 soft-RDMA plus the native HCA),
stock udev rules can rename RDMA devices at boot and race your setup. On the
test rig the IB device was udev-named after its netdev (`ibp195s0`). Pin the
name you want, or disable persistent naming for the IB device:

```bash
# Example: keep the kernel's deterministic per-rail name
sudo touch /etc/udev/rules.d/60-rdma-persistent-naming.rules.d/99-keep-native-ib.conf
```

An empty override of `60-rdma-persistent-naming.rules` achieves the same in
other setups — keep the name you pin stable across reboots.

## 3. Verify the fabric from the host

```bash
ibv_devinfo | grep -E 'hca_id|state|port|active_speed|active_width'
ibstat
rdma link
```

Both nodes should show the HCA port as `ACTIVE`, `LINK_UP`, with `active_speed:
40 Gb/s` (or the negotiated rate). Then confirm the container can see the HCA —
the `refresh_toolbox.sh` script does this automatically, but you can check with:

```bash
ls /dev/infiniband
```

## 4. Selecting the InfiniBand transport

1.  Enter the toolbox and run the cluster manager:
    ```bash
    start-vllm-cluster
    ```
2.  Configure the Head/Worker IPs (the **Ethernet** management IPs — Ray and
    GLOO rendezvous, not the IB fabric).
3.  Select **"Start Ray Cluster"**, then **"Select Transport"** and choose
    **`infiniband`** (or leave **`auto`** — it detects an active native-IB HCA
    and selects InfiniBand automatically; otherwise it falls back to RDMA/RoCE,
    then Ethernet).
4.  Start the cluster as usual; both the Ray head/worker setup **and** the
    subsequent vLLM launch read the same choice.

## 5. What the launcher sets (and why)

For `infiniband`, `scripts/cluster_manager.py` exports:

| Variable | Value | Why |
| :--- | :--- | :--- |
| `NCCL_IB_DISABLE` | `0` | Enable the RDMA data path. |
| `NCCL_IB_HCA` | one detected `<device>:<port>` | Pin a **single** HCA. An ambiguous prefix matching several ports makes RCCL's `ncclCommInitRank` fail with *"internal error"*. |
| `NCCL_IB_GID_INDEX` | `0` | Native IB uses the **link-local GID**. Index `1` is RoCEv2-IPv4-only and does not exist on an IB fabric. |
| `NCCL_PROTO` | `LL` | Low-latency protocol for the latency-bound decode all-reduce. |
| `NCCL_ALGO` | `Ring` | Matches the validated TP=2 profile. |
| `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT` | `23` / `7` | Fabric stability on the IB link. |
| `NCCL_NET_GDR_LEVEL` | `0` | gfx1151 has no GPUDirect — RCCL host-stages the prefill all-reduce. Expected, not a fault. |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | management iface | Control-plane sockets stay on Ethernet; only the data plane uses IB. |

The HCA is detected per-node inside the generated setup script (first active
`/sys/class/infiniband/*/ports/*/state`), so device names never have to match
across hosts. To force a specific device, set **`VLLM_IB_HCA=<device>:<port>`**
in the environment before starting the cluster — it wins over detection for
both the setup scripts and the vLLM launch.

When several active RDMA ports exist, the launcher prints a warning listing all
candidates plus the effective HCA, so a silently-picked rail is always visible
(set `VLLM_IB_HCA` to choose another).

## 6. Troubleshooting

### `ncclCommInitRank` fails with "internal error"
RCCL could not resolve `NCCL_IB_HCA` to exactly one device. The launcher pins one
port, but if you have multiple HCAs and overrode `NCCL_IB_HCA` manually, narrow it
to a single `<device>:<port>`.

### Ports stuck at `INIT` / not `ACTIVE`
No (or conflicting) Subnet Manager. Confirm `opensm` is running on exactly one
node and both cables are seated.

### Slow prefill all-reduce
gfx1151 has no GPUDirect — large (>1 MiB) all-reduces are host-staged through
`NCCL_NET_GDR_LEVEL=0`. This is expected; only decode's latency-bound small
all-reduces see the full IB benefit.

### RCCL logging
To debug an init failure, temporarily enable logging (the launcher keeps it off
to reduce journal noise):

```bash
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET
```

Expect the harmless *"GPU Direct RDMA not available for device 0"* line on
gfx1151.

---

This setup was tested with ConnectX-3 (mlx4) InfiniBand adapters.
