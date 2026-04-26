\
// RUN: ts-spectral-opt --tessera-spectral-distributed %s | FileCheck %s --check-prefix=DIST
// DIST: // TODO: expect CommQ all-to-all constructs
