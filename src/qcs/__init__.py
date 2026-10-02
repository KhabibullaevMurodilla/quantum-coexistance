"""
qcs -- Quantum Coexistence Scheduler
=====================================
A network-level simulator and scheduler testbed for entanglement-based
quantum networks sharing resources (optical fiber links, time slots) with
competing applications and, optionally, classical DWDM traffic.

Scope: this package deliberately works at the level most of the network
utility maximization (NUM) / scheduling literature uses -- links
characterized by a Werner-state parameter governing a rate/fidelity
tradeoff, sessions requesting entanglement at some rate and fidelity floor,
under a coherence-time cutoff. It is NOT a physical-layer quantum optics
simulator (that's what SeQUeNCe / NetSquid are for); this operates one
level up, which is the right level for studying scheduling and resource
allocation policy, not device physics.
"""

__version__ = "0.1.0"
