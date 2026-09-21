# Tessera Empirical Search Pass (stub)

Registers `-tessera-empirical-search` in tessera-opt. The pass is a thin shim that
marshals IR snapshots to the Python runner (or an in-proc C++ agent) and receives
score annotations for IR variants.
