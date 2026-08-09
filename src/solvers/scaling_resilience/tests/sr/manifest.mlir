// RUN: tessera-opt -tessera-export-deployment-manifest %s | FileCheck %s --check-prefix=MNF
tessera.mesh.create { axis_names = ["data","model"] }
tessera_collective.all_reduce { mesh_axis = "data", reduction = "sum" }
tessera_sr.checkpoint { }
// MNF: export
