\
// RUN: ts-spectral-opt --tessera-legalize-spectral %s | FileCheck %s --check-prefix=LEGAL
// LEGAL: // TODO: expect Tile IR staging ops after legalization
