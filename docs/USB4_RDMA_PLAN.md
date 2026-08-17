# USB4 RDMA integration plan

Status: **planning and handoff document; not yet implemented**

Last host inspection: **2026-08-17**

Target repository: `kyuz0/amd-strix-halo-vllm-toolboxes`

This document contains the context needed to add AlexKGwyn's experimental
Thunderbolt/USB4 RDMA stack to this toolbox without losing the already-working
Intel E810 RoCE path. It is intended to be sufficient for a new engineer or
LLM to resume the work without relying on chat history.

> [!CAUTION]
> The host portion replaces the running kernel's stock `thunderbolt` and
> `thunderbolt_net` modules with a matched patched pair. Alex's documentation
> warns that a mismatched pair can panic the host when the cable connects.
> Never live-reload the Thunderbolt core and never install or reload only one
> side of the two-host link. Host installation requires explicit user approval,
> a coordinated maintenance window, and a coordinated reboot of both hosts.

No host modules were installed and no host configuration was changed while
preparing this plan. The observations below came from read-only SSH commands.

## Objective

Support these independent cluster transports from `start-vllm-cluster`:

| Profile | Data/control link | RCCL transport | Intended use |
|---|---|---|---|
| E810 RoCE | `enp194s0np0` | RDMA through `rocep194s0` | Existing high-bandwidth path |
| E810 TCP | `enp194s0np0` | Sockets | Existing diagnostic fallback |
| USB4 RDMA | `thunderbolt0` | RDMA through `usb4_rdma0` | New generic RCCL path |
| USB4 TCP | `thunderbolt0` | Sockets | USB4 link and fallback validation |

Generic RCCL over the USB4 verbs device is the first objective. Alex's custom
`tbv_ar2` small-collective implementation is a separate, optional performance
phase. Do not make the vLLM patch a prerequisite for validating the host
driver, userspace provider, or generic RCCL.

## Inspected upstream baseline

The analysis used
[`AlexKGwyn/ds4-vllm-public`](https://github.com/AlexKGwyn/ds4-vllm-public)
at exact commit
[`95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff`](https://github.com/AlexKGwyn/ds4-vllm-public/tree/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff).
Review that exact tree before using a newer revision; do not silently mix files
from different revisions.

Important upstream paths:

- [top-level runbook](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/AGENTS.md)
- [TBV/USB4 RDMA explanation](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/tbv/README.md)
- [module builder](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/tbv/build-modules.sh)
- [module and boot integration installer](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/tbv/install-modules.sh)
- [verbs device bring-up](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/tbv/bringup/tbv-reload-roce.sh)
- [container provider/native build](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/container/Dockerfile)
- [`tbv_ar2` native implementation](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/container/native/tbv_ar2.hip)
- [upstream patch inventory](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/container/patches/MANIFEST.md)
- [third-party notices](https://github.com/AlexKGwyn/ds4-vllm-public/blob/95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff/THIRD_PARTY_NOTICES.md)

Upstream host source pins at that commit:

| Component | Pin | Purpose |
|---|---|---|
| `westeri/thunderbolt` | `503c5ae1e72aa9ed91925dafa3d82ee2e992747f` | Patched Thunderbolt core and net sources |
| `hellas-ai/thunderbolt-ibverbs` | `76ba39b630a70accb72f19388eefe48844b50eb8` | Out-of-tree verbs driver, kernel patch series, and rdma-core provider patches |
| Alex local `ibverbs-local.patch` | From the inspected commit | RC-write/zero-copy additions on the pinned verbs driver |

Alex's repository is Apache-2.0 for its original code. The kernel components
and derivative kernel patches are GPL-2.0, while rdma-core uses GPL-2.0 OR
Linux-OpenIB. Preserve the upstream fetch-at-build strategy, pins, notices,
patch provenance, and applicable license texts. Do not copy compiled modules or
provider binaries into this repository.

## Verified fw1/fw2 state

These are point-in-time observations, not durable configuration guarantees.
Re-run the preflight before building or installing anything.

`fw1` and `fw2` are SSH aliases. `fw1` reports the static hostname
`frmwk-dsk`; `fw2` reports `fw2`.

| Property | fw1 | fw2 |
|---|---|---|
| Hardware | Framework Desktop, AMD Ryzen AI Max 300 / gfx1151 | Framework Desktop, AMD Ryzen AI Max 300 / gfx1151 |
| OS | Fedora Linux 43 Workstation | Fedora Linux 43 Workstation |
| Running kernel | `7.1.3-100.fc43.x86_64` | `7.1.3-100.fc43.x86_64` |
| Kernel build tree | `/usr/src/kernels/7.1.3-100.fc43.x86_64` | Same |
| Secure Boot | Disabled | Disabled; platform also reports Setup Mode |
| USB4 controllers | AMD Strix Halo USB4 host routers using stock `thunderbolt` | Same |
| Loaded USB4 modules | Stock `thunderbolt`, `thunderbolt_net` | Same |
| Custom TBV modules | Not present | Not present |
| Current RDMA device | `rocep194s0`, active/link-up | `rocep194s0`, active/link-up |
| Current USB4 RDMA device | None | None |

Current network layout:

| Network | fw1 | fw2 | Notes |
|---|---|---|---|
| Management LAN | `enp191s0`, `192.168.1.128/24` | `enp191s0`, `192.168.1.127/24` | Prefer the SSH aliases for management |
| E810/RoCE | `enp194s0np0`, `192.168.100.1/30` | `enp194s0np0`, `192.168.100.2/30` | Existing working RCCL/RDMA path |
| USB4 IP | `thunderbolt0`, `192.168.2.1/24` | `thunderbolt0`, `192.168.2.2/24` | Existing direct cable |

The USB4 link was verified as follows:

- `thunderbolt0` was `UP,LOWER_UP` on both hosts.
- MTU was 9000 on both hosts.
- sysfs reported 20.0 Gb/s receive and transmit speed on both ends.
- Each endpoint identified the other Linux host.
- Three ICMP probes succeeded in each direction with no loss. The observed
  averages were approximately 0.70 ms from fw1 and 0.24 ms from fw2; this is
  only a connectivity check, not an RDMA or stable latency benchmark.
- NetworkManager already has manual `thunderbolt0` profiles using
  `192.168.2.1/24` and `192.168.2.2/24`.
- `/dev/infiniband` exists on both hosts for the E810 device.

The fw2 kernel configuration exposed `CONFIG_USB4=m`, `CONFIG_USB4_NET=m`, and
the expected InfiniBand/RDMA options. The equivalent `/boot/config` was not
readable on fw1 during inspection, although fw1 is running the same kernel and
its stock USB4 and RDMA devices are operational. Do not treat that as proof
that a patched module will build: the no-sudo module build is the real gate.

## Current local implementation

The existing image and launcher are designed around the E810 path:

- [`Dockerfile.ubuntu-repoamd`](../Dockerfile.ubuntu-repoamd) builds and overlays
  rdma-core `v62.0`. This was introduced because Ubuntu's rdma-core 50 provider
  generated RCCL `Unknown completion` failures with the host `irdma` driver.
  The image must retain the complete matched v62 userspace stack and its
  `libirdma` provider.
- [`refresh_toolbox.sh`](../refresh_toolbox.sh) already exposes
  `/dev/infiniband` and sets unlimited memlock when the host has RDMA devices.
  This should also expose the USB4 device after host bring-up because it maps
  the directory, not a single HCA.
- [`scripts/cluster_manager.py`](../scripts/cluster_manager.py) currently
  derives one interface from the selected IP subnet and exports only generic
  socket/Gloo/RCCL settings. It does not select an HCA explicitly.
- [`scripts/start_vllm_cluster.py`](../scripts/start_vllm_cluster.py) currently
  offers a boolean "Force Ethernet" choice. It re-derives the interface again
  when serving and hardcodes GID index 1 and GDR level 0.
- [`VLLM_PATCH_MANIFEST.md`](VLLM_PATCH_MANIFEST.md) is the maintenance ledger
  for local vLLM, ROCm, and rdma-core changes. Every imported USB4/provider or
  custom-all-reduce patch needs an entry there.
- [`rdma_cluster/setup_guide.md`](../rdma_cluster/setup_guide.md) currently
  documents ordinary Thunderbolt TCP/IP separately from E810 RDMA. It must not
  imply that stock `thunderbolt_net` is itself an RDMA device.

The validated vLLM development baseline currently recorded by the manifest is
`79f3183f86b89c3bda05d467041bf3ef9ef60426`, whereas Alex's monolithic vLLM
patch targets `470229c`. Never apply Alex's entire vLLM patch to the current
tree. Only re-port the narrow custom communicator hook if the optional
`tbv_ar2` phase is reached.

## Architecture to preserve

There are three mandatory layers and one optional layer:

1. **Host USB4/RDMA stack:** matched patched `thunderbolt`,
   `thunderbolt_net`, `thunderbolt_ibverbs`, and `nhi_throttle` modules plus
   controlled boot ordering.
2. **Container userspace:** a `usb4_rdma` libibverbs provider built against the
   same rdma-core v62 ABI already used by this image.
3. **Cluster orchestration:** an explicit transport profile propagated to the
   head Ray daemon, worker Ray daemon, and vLLM driver.
4. **Optional custom all-reduce:** the vLLM hook, Python wrappers, and
   `libtbv_ar2.so`, gated behind an explicit USB4-only feature flag.

Do not collapse these into one patch. Each layer needs an independent
validation and rollback point.

## Non-negotiable adaptations

### Keep the two physical networks separate

Alex's deployment uses `192.168.100.1/2` on `thunderbolt0`. That conflicts
with the already-configured E810 link here. This repository must use:

- E810: `192.168.100.1` and `192.168.100.2`
- USB4: `192.168.2.1` and `192.168.2.2`

Alex's bring-up script waits specifically for `192.168.100.*`, and the custom
all-reduce wrappers hardcode `192.168.100.1`. Both assumptions must be removed.

### Preserve both RDMA devices

Alex supplies an empty `/etc/udev/rules.d/60-rdma-persistent-naming.rules`
override because USB4 is the only HCA in that deployment. Installing that
unchanged here could interfere with the E810 name `rocep194s0`.

Use either a targeted udev rule matching only the `thunderbolt_ibverbs` driver
or a deterministic rename in the USB4 service after identifying the device by
driver/sysfs parent. It must produce a stable configured name such as
`usb4_rdma0` without touching `rocep194s0`.

RCCL must receive one exact HCA name:

```text
NCCL_IB_HCA=usb4_rdma0
```

Do not use the prefix `usb4_rdma`; Alex documents that multiple prefix matches
can cause `ncclCommInitRank` internal errors after link resets.

### Keep rdma-core v62

Alex builds the provider by patching rdma-core v57 because that matches his
older Fedora-based image. This image uses Ubuntu library paths and rdma-core
v62. Downgrading or mixing providers would risk reintroducing the already-fixed
E810 completion failure.

The implementation must first test whether the pinned provider patch series
applies cleanly and compiles against rdma-core v62. Treat any failed patch or
changed provider API as a compatibility investigation, not something to bypass
with fuzzy patching. Build and install the actual provider ABI filename emitted
by v62 rather than guessing an `rdmavNN` suffix.

### Separate management and cluster addresses

The current launcher uses `worker_ip` both as the Ray address and SSH target.
That is fragile when selecting USB4: a broken data cable would also prevent the
launcher from stopping or diagnosing the remote Ray process.

Represent at least these values independently:

```yaml
worker_ssh_host: fw2
head_cluster_ip: 192.168.2.1
worker_cluster_ip: 192.168.2.2
socket_iface: thunderbolt0
rdma_hca: usb4_rdma0
gid_index: 1
```

### Do not confuse generic GDR with the custom path

The current generic RCCL configuration uses `NCCL_NET_GDR_LEVEL=0` on gfx1151.
Alex's custom driver and `tbv_ar` path have separate dma-buf registration
behavior. That does not mean generic RCCL GPUDirect should be enabled. Keep GDR
disabled for the first USB4 RCCL validation and change it only after a focused,
measured correctness test.

## Proposed repository changes

### 1. Host support under `host/usb4-rdma/`

Import or adapt the upstream build/bring-up logic with provenance headers and
the exact source pins. The directory should contain:

- `README.md`: host-specific runbook and safety warnings;
- `build-modules.sh`: no-sudo build into a per-kernel cache;
- `verify-build.sh`: verify all four artifacts, vermagic, source pins, expected
  marker strings, and hashes;
- `install-modules.sh`: explicit root-only installation with backups and
  fail-closed checks;
- `uninstall-modules.sh`: remove units/blacklists/kernel arguments and restore
  stock behavior on the next coordinated reboot;
- `status.sh`: read-only report for module versions, services, link, HCA, GIDs,
  provider visibility, and memlock;
- `bringup/`: parameterized one-cable bring-up, targeted naming, and memlock;
- `systemd/`: matched-core loading and USB4 verbs bring-up services;
- `nhi-throttle-mod/`: the small GPL helper with its license/provenance;
- patch files or fetch instructions needed to reproduce the pinned composition.

The initial scope is **one cable**. Alex states that one cable carries IP and
RDMA together, leaves transmit zero-copy active, and disables the separate
receive zero-copy rail because the NHI has no spare rings. His two-cable mode
is substantially more fragile and is out of scope until the one-cable path is
stable.

`nhi_throttle` changes NHI interrupt moderation from the stock 128 us setting
to 8 us. Treat it as an independently testable performance component. Bring up
correct RDMA first; compare latency with and without the throttle before making
it mandatory.

### 2. Container provider in `Dockerfile.ubuntu-repoamd`

Extend the existing rdma-core builder stage rather than creating a second,
incompatible libibverbs installation:

1. Fetch rdma-core `v62.0` as today.
2. Fetch `hellas-ai/thunderbolt-ibverbs` at the pinned commit.
3. Apply only its userspace provider patch series to the v62 tree.
4. Build the complete v62 tree once.
5. Install the resulting `usb4_rdma` provider and driver registration together
   with the existing v62 `libibverbs` and `libirdma` artifacts.
6. Assert that both the E810 and USB4 provider artifacts exist, run `ldconfig`,
   and inspect dependencies at image-build time.

The target image uses `/usr/lib/x86_64-linux-gnu`, unlike Alex's Fedora image
which copies into `/usr/lib64`. Follow the actual CMake install paths from the
Ubuntu build.

`refresh_toolbox.sh` probably needs only clearer diagnostics, because its
current whole-directory `/dev/infiniband` mapping is appropriate. Confirm this
with the real `usb4_rdma0` device after host installation.

### 3. Transport profiles in the launcher

Add a small transport definition module rather than spreading new environment
conditionals across both launch files. A profile should contain:

- display name and stable key;
- head and worker cluster IP defaults;
- worker SSH target default;
- socket/Gloo interface;
- exact HCA or no HCA;
- GID index;
- `NCCL_IB_DISABLE` and `NCCL_NET_GDR_LEVEL`;
- timeout/retry defaults;
- whether USB4 host/provider preflight is required.

Suggested defaults:

| Key | Head/worker IPs | Interface | HCA | `NCCL_IB_DISABLE` |
|---|---|---|---|---:|
| `e810_rdma` | `192.168.100.1/2` | `enp194s0np0` | `rocep194s0` | 0 |
| `e810_tcp` | `192.168.100.1/2` | `enp194s0np0` | unset | 1 |
| `usb4_rdma` | `192.168.2.1/2` | `thunderbolt0` | `usb4_rdma0` | 0 |
| `usb4_tcp` | `192.168.2.1/2` | `thunderbolt0` | unset | 1 |

Make "Select Cluster Transport" a persistent top-level TUI option alongside
"Select Target Toolbox". Start, stop, status, launch, environment display, and
preflight must all use the selected profile. The same environment must reach:

- the local Ray head daemon;
- the remote Ray worker daemon inside the selected toolbox;
- the vLLM driver that creates the distributed workers.

Do not infer USB4 from a subnet or select the first RDMA device. Explicitness is
required because both links and both HCAs will coexist.

### 4. Preflight before Ray

For both hosts, and inside the selected head/worker toolboxes where relevant,
verify:

1. the configured interface exists, is up, and owns the configured IP;
2. the peer cluster address responds;
3. the configured HCA exists exactly once when RDMA is enabled;
4. `rdma link show` reports the expected active link;
5. the configured GID index is present and non-zero;
6. `ibv_devices` can open the HCA inside the toolbox;
7. `/dev/infiniband` is mapped and memlock is unlimited;
8. the `usb4_rdma` provider registration exists for the USB4 profile;
9. `rocep194s0` still exists after USB4 installation;
10. head and worker profile values agree.

Fail before starting Ray if any required check fails. Print the exact failed
check and corrective command; never silently fall back from USB4 RDMA to TCP.

## Host implementation and deployment gates

### Gate A: build only, no sudo

Run the adapted builder against `7.1.3-100.fc43.x86_64` on both hosts. It should
produce:

- `thunderbolt-patched.ko`
- `thunderbolt_net.ko`
- `thunderbolt_ibverbs.ko`
- `nhi_throttle.ko`

Verify identical source pins, expected marker strings, and exact vermagic.
Because the two machines currently run the same kernel, compare artifact
hashes; investigate differences before proceeding. Successful compilation is
not permission to install.

### Gate B: container build and provider inspection

Build a development image containing both v62 providers. Before host changes,
verify the USB4 provider's ABI and loader registration statically. Continue to
run the existing E810 two-rank RCCL test with that image so the new provider
cannot regress the current path.

### Gate C: coordinated host installation

This is the first destructive/privileged phase and needs explicit approval.
Before it:

- stop vLLM and Ray cleanly on both hosts;
- record current kernel, kernel arguments, loaded modules, NetworkManager
  profiles, active RDMA devices, and service state;
- ensure out-of-band/management SSH works independently of `thunderbolt0`;
- ensure the uninstall/rollback path has been reviewed;
- keep the single USB4 cable topology;
- install the complete matched set on both hosts before rebooting either;
- reboot both hosts together.

Never attempt to switch the core live. Alex reports that live or staggered
reloads can wedge the Thunderbolt HopID/tunnel allocator with errors such as
`failed to allocate Rx HopID` or `native tunnel enable failed ... EBUSY`.
Recovery may require rebooting both hosts together or disconnecting and
reconnecting the cable.

Kernel modules are vermagic-locked. After every kernel update, rebuild and
verify all four modules for the new kernel before booting the patched stack.
The status tooling should detect a running-kernel/staged-module mismatch and
report it loudly rather than allowing a silent TCP fallback.

## Validation ladder

Do not start with DeepSeek or `tbv_ar2`. Validate from the bottom up:

1. **Post-boot host state:** patched matched core/net loaded on both machines;
   `thunderbolt0` still has `192.168.2.1/2`; E810 remains active.
2. **USB4 HCA state:** one stable `usb4_rdma0` per host; active link; expected
   non-zero GID at the configured index.
3. **Container visibility:** both selected toolboxes see `usb4_rdma0` through
   `ibv_devices`; `rocep194s0` remains visible.
4. **Verbs correctness:** `ib_write_lat` and `ib_write_bw` across a range of
   message sizes, followed by repeated setup/teardown cycles.
5. **Generic RCCL:** the existing two-rank PyTorch all-reduce probe over USB4,
   testing at least 1 KiB through 16 MiB, synchronizing and destroying the
   process group explicitly, and checking logs for the exact HCA.
6. **Ray:** start and stop a two-node Ray cluster repeatedly with the USB4
   profile; verify advertised node IPs and GPU resources.
7. **Small TP=2 model:** load, infer, stop, and restart before using DeepSeek.
8. **DeepSeek V4:** startup, fixed-output correctness, long-context recall,
   concurrency, and sustained benchmark runs.
9. **Comparison:** record E810 TCP, E810 RoCE, USB4 TCP, and USB4 RDMA latency,
   bandwidth, prefill, decode, stability, CPU use, and GPU use.

The live cable negotiated 20 Gb/s. Do not assume USB4 RDMA will beat the E810
for large collectives. Its potential value may be lower small-message latency,
especially with the optional custom all-reduce. Measure rather than infer.

## Optional `tbv_ar2` phase

Alex's repository contains two custom two-rank all-reduces:

- `tbv_ar` v1: direct verbs path with optional GPU dma-buf memory registration;
- `tbv_ar2` v2: pinned host staging, a progress thread, and GPU polling/addition.

At the inspected revision, `tbv_ar2` is limited by its wrappers/native code to
two ranks, contiguous BF16/FP16/FP32 tensors, and messages no larger than 1 MiB.
It uses GID index 1 and defaults to port 18531; v1 defaults to 18515. The Python
wrappers hardcode the head address `192.168.100.1`, and the native code selects
an HCA by the `usb4_rdma` prefix. Both must be parameterized.

Alex's source comments report roughly 105 us for v2 versus 228 us for v1 and a
slower RCCL path in that deployment. Those are upstream-reported measurements,
not results verified on fw1/fw2 or against the current image.

If ported:

1. isolate the communicator hook from Alex's monolithic vLLM patch;
2. rebase it against the exact current vLLM commit and current communicator
   lifecycle;
3. configure peer address, exact HCA, GID and ports through the selected
   transport profile;
4. enable it only for TP=2 USB4 RDMA;
5. retain the normal RCCL chain for initialization failures, unsupported
   tensors, payloads above 1 MiB, prefill, and graph-capture/ineligible paths;
6. log selection and fallback loudly enough that a benchmark cannot silently
   measure RCCL while claiming `tbv_ar2`;
7. test exact numerical agreement, repeated sequence/slot reuse, timeouts,
   process teardown, both rank orderings, and injected peer failure.

Add its Python wrappers and native libraries only after generic USB4 RCCL is
stable. It should be a separate environment gate and patch-manifest entry with
an explicit removal/review condition.

## Tests to add

Suggested local test coverage:

- transport profile schema/defaults;
- exact environment produced for every profile;
- identical propagation to head, worker, and vLLM driver;
- separation of SSH host from Ray IP;
- exact-HCA enforcement and no prefix matching;
- USB4 and E810 addresses never overlap;
- preflight parsing for missing device, duplicate HCA, zero GID, provider not
  loaded, link down, memlock failure, and peer unreachable;
- generated remote shell content uses the chosen profile and selected toolbox;
- stop/status still use the management target if the data interface is down;
- Dockerfile source pins and assertions for both providers;
- patch application fails closed on rdma-core or vLLM source drift;
- optional `tbv_ar2` eligibility, fallback, and environment tests.

Run the repository's normal Python unit suite, Python compilation checks, and
`git diff --check`, then build only the development image. A successful image
build is not host or GPU validation.

## Rollback design

Rollback must be implemented and reviewed before host installation. It should:

1. disable the TBV systemd services;
2. remove only TBV-owned module staging files;
3. remove the targeted USB4 RDMA naming rule;
4. restore any backed-up memlock, NetworkManager, modprobe, and systemd files;
5. remove only the kernel arguments added by the installer;
6. regenerate initramfs if the installation changed it;
7. leave E810 configuration untouched;
8. reboot both hosts together into the stock `thunderbolt`/`thunderbolt_net`
   stack;
9. verify `thunderbolt0` TCP and `rocep194s0` after rollback.

Do not define rollback as unloading the patched core on a live cable.

The container rollback is simpler: select the previous image/tag or use the
E810 transport profile. The USB4 provider should remain inert when no
`usb4_rdma*` device exists.

## Acceptance criteria

The initial USB4 RDMA feature is complete only when:

- host modules build reproducibly for the running kernel on both machines;
- install and rollback procedures are documented and tested deliberately;
- both E810 and USB4 RDMA devices coexist with stable, distinct names;
- the toolbox sees and opens both providers;
- generic two-rank RCCL over `usb4_rdma0` passes repeated correctness and
  teardown tests without falling back to sockets;
- `start-vllm-cluster` can select any of the four profiles and uses it
  consistently for Ray and vLLM;
- USB4 link failure produces an actionable preflight failure, not a hang or
  silent fallback;
- TP=2 inference and the relevant DeepSeek recall/concurrency benchmark pass;
- existing E810 TCP/RDMA behavior remains unchanged;
- the README, RDMA guide, and patch manifest describe the experimental status,
  kernel coupling, source pins, and maintenance requirements.

The optional `tbv_ar2` work has separate acceptance criteria and must not be
used to declare the generic USB4 RDMA integration complete.

## Open questions to resolve during implementation

1. Do the pinned USB4 provider patches apply cleanly to rdma-core v62, and what
   provider ABI filename does that build generate?
2. Do the pinned Thunderbolt and verbs modules compile unchanged against Fedora
   kernel `7.1.3-100.fc43.x86_64`?
3. What is the most reliable targeted rule/sysfs match for naming only the
   `thunderbolt_ibverbs` HCA?
4. Is GID index 1 consistently populated after the adapted bring-up on both
   hosts?
5. Does generic RCCL use the selected USB4 HCA without unknown completions,
   hangs, or teardown errors across repeated runs?
6. Does `nhi_throttle` provide a material benefit on these hosts, and does it
   affect stability or power behavior?
7. Does USB4 improve the small all-reduce shapes that limit TP=2 decode, or is
   the current E810 path already faster overall?
8. Is `tbv_ar2` still compatible with the current vLLM communicator lifecycle,
   and does it improve end-to-end decode rather than only a component timing?

## Pickup checklist for a new agent

1. Read this entire document.
2. Inspect the current worktree and preserve unrelated/untracked user files.
3. Read the current versions of the local files linked under "Current local
   implementation"; line numbers and pins may have changed.
4. Fetch or inspect Alex's repository at the exact recorded commit and read its
   `AGENTS.md`, `tbv/README.md`, build scripts, container provider build, patch
   manifest, and third-party notices.
5. Re-run the read-only fw1/fw2 preflight because kernel/network state can
   change.
6. Implement host build-only support, the v62 container provider, transport
   profiles, preflight, tests, docs, and rollback before requesting permission
   for a host installation.
7. Never install modules, alter kernel arguments, restart the hosts, unplug the
   cable, or stop a running cluster without explicit user authorization.
