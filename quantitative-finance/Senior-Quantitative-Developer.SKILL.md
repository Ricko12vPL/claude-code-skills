---
name: senior-quantitative-developer
description: Build low-latency trading infrastructure. Use when optimizing market data, execution systems, or reducing latency.
---

# Senior Quantitative Developer

Production-grade guidance for building and improving trading systems end-to-end.

## Instructions

### When to Use
- Build/optimize market data, order routing, execution, and risk modules
- Reduce latency/jitter; increase throughput and reliability
- Harden production: observability, CI/CD, rollback, incident response

### Expected Outcomes
- Latency budgets (P50/P95/P99) for tick→signal→order met and monitored
- Zero data-loss with bounded backpressure; deterministic recovery/replay
- SLO compliance with low MTTR and safe deployments
- Efficient CPU/cache/NUMA usage and infra cost per notional lowered

### Required Inputs
- Venue/protocol specs (FIX/ITCH/OUCH/native), product set, time-sync (PTP)
- Baseline metrics (latency, drop rate, CPU/mem, NIC stats) and failure modes
- Risk/Compliance constraints (pre-trade checks, throttles, kill-switch, audit)
- Data schemas (ticks/books/trades), sequencing, recovery/replay policies

### Implementation Steps
1) Define non-functional targets (latency SLOs, loss budgets, load envelopes)
2) Establish perf harness (synthetic feeds, pcap replay, determinism, HW counters)
3) Optimize hot path:
   - C++: lock-free structures, pre-allocation, cache-aware layouts, SIMD, zero-copy I/O
   - Networking: tuned NIC/IRQ/RSS, busy-poll where appropriate, kernel bypass if needed
   - Storage/queues: ring buffers, memory arenas, avoid heap on critical path
4) Correctness & resilience: sequence gaps, idempotency, bounded backpressure, circuit breakers
5) Observability: structured logs, traces, RED/USE metrics, per-hop latency histograms
6) CI/CD: hermetic builds (Bazel/CMake), canary/blue-green, feature flags, rollback
7) Incident response: SLO alerts, runbooks, automatic capture (pcap/core), postmortems

### Quality Checklist
- [ ] No heap allocations on hot path; ownership explicit; false sharing avoided
- [ ] Monotonic timestamps and verified PTP; end-to-end timing captured
- [ ] Gap/recovery logic tested; replay deterministic; idempotent handlers
- [ ] Pre-/post-trade risk enforced; throttles and kill-switch verified
- [ ] Canary criteria and automatic rollback defined
- [ ] Perf tests at peak load + failure + recovery scenarios

### Metrics & Validation
- Latency (P50/P95/P99), jitter; tail amplification across components
- Loss: packet drop rate, sequence gaps and recovery success
- Throughput: msgs/sec sustained; CPU cycles/instruction; cache miss rates
- Reliability: SLOs, MTTR, incident/change failure rates, rollback rate

## Tools & Technologies
- Languages: C++20/23, Python (tooling); optionally Rust/Java
- Build/CI: Bazel/CMake, Conan/vcpkg, GitHub Actions/Jenkins
- Perf: perf, VTune, FlameGraph, BPF/bcc, hwloc
- Networking: DPDK/XDP (where applicable), ethtool, tc; IRQ/RSS affinity tuning
- Data: kdb+/q, ClickHouse, PostgreSQL; Kafka/NATS; Parquet/Arrow
- Observability: Prometheus/Grafana, OpenTelemetry, ELK

## Examples
- "Zredukuj P99 latencji feed-handlera do < X µs i pokaż flamegraph przed/po."
- "Zaimplementuj bounded backpressure i test przeciążenia na 2× peak volume."
- "Skonfiguruj PTP i end-to-end timestamping; raport dryfu i offsetów."

### Common Pitfalls
- Skryte alokacje i contention; false sharing; nieoptymalna NUMA lokalność
- Błędna synchronizacja czasu → zafałszowane TCA i analizy ryzyka
- Brak deterministycznego recovery/replay i idempotencji

## References
- [Citadel Securities – Senior Quantitative Developer](https://www.citadelsecurities.com/careers/details/senior-quantitative-developer/)
- [Hudson River Trading – Careers](https://www.hudsonrivertrading.com/careers/)
- [Jump Trading – Careers](https://www.jumptrading.com/careers/)
- [Jane Street – Software Engineering](https://www.janestreet.com/join-jane-street/)
- [Two Sigma – Engineering](https://www.twosigma.com/careers/)
- [Anthropic – Skills (Claude)](https://www.anthropic.com/news/skills)


