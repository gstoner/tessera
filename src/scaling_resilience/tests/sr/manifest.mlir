// RUN: tessera-opt -tessera-export-deployment-manifest %s | FileCheck %s --check-prefix=MNF
tessera.mesh.create { axis_names = ["data","model"] }
tessera.collective.all_reduce { axis = "data", op = "sum" }
tessera_sr.checkpoint { }
// MNF: export