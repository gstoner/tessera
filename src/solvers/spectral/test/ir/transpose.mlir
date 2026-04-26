\
// RUN: ts-spectral-opt --tessera-spectral-transpose-plan %s | FileCheck %s --check-prefix=TP
// TP: // TODO: expect planned transposes
