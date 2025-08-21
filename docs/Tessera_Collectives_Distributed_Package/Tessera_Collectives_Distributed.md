# Tessera Collectives & Distributed Systems Guide

This guide provides normative definitions and best practices for distributed execution in Tessera.

## Mesh Layouts and Parallelism

Tessera supports **data parallel (DP)**, **tensor parallel (TP)**, and **pipeline parallel (PP)** layouts.  
See figure below:

![Tessera Mesh Collectives](images/mesh_collectives.png)

## ZeRO-Flow Integration

Optimizer state partitioning across devices with checkpointing.  
See figure below:

![ZeRO-Flow State Partitioning](images/zero_flow.png)
