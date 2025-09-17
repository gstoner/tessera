// RUN: tessera-opt-sr -tessera-export-deployment-manifest %s | FileCheck %s
tessera_sr.export_manifest { path = "manifest.json", include = ["mesh","optimizer"] }
// CHECK: export_manifest