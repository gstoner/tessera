# 16.20.3. Global Instructions

> RDNA3.5 ISA — pages 640–653

SCRATCH_STORE_D16_HI_B16                                                                                           37

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

  MEM[ADDR].b16 = VDATA[31 : 16]

16.20.3. Global Instructions
Global instructions are like Flat, but assume all work-item addresses fall in global memory space.

GLOBAL_LOAD_U8                                                                                                     16

Load 8 bits of unsigned data from the global aperture, zero extend to 32 bits and store the result into a vector
register.

  VDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })

GLOBAL_LOAD_I8                                                                                                     17

Load 8 bits of signed data from the global aperture, sign extend to 32 bits and store the result into a vector
register.

  VDATA.i32 = 32'I(signext(MEM[ADDR].i8))

GLOBAL_LOAD_U16                                                                                                    18

Load 16 bits of unsigned data from the global aperture, zero extend to 32 bits and store the result into a vector
register.

  VDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })

GLOBAL_LOAD_I16                                                                                                    19

Load 16 bits of signed data from the global aperture, sign extend to 32 bits and store the result into a vector
register.

  VDATA.i32 = 32'I(signext(MEM[ADDR].i16))

GLOBAL_LOAD_B32                                                                 20

Load 32 bits of data from the global aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32

GLOBAL_LOAD_B64                                                                 21

Load 64 bits of data from the global aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32

GLOBAL_LOAD_B96                                                                 22

Load 96 bits of data from the global aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32;
  VDATA[95 : 64] = MEM[ADDR + 8U].b32

GLOBAL_LOAD_B128                                                                23

Load 128 bits of data from the global aperture into a vector register.

  VDATA[31 : 0] = MEM[ADDR].b32;
  VDATA[63 : 32] = MEM[ADDR + 4U].b32;
  VDATA[95 : 64] = MEM[ADDR + 8U].b32;
  VDATA[127 : 96] = MEM[ADDR + 12U].b32

GLOBAL_STORE_B8                                                                 24

Store 8 bits of data from a vector register into the global aperture.

  MEM[ADDR].b8 = VDATA[7 : 0]

GLOBAL_STORE_B16                                                                      25

Store 16 bits of data from a vector register into the global aperture.

  MEM[ADDR].b16 = VDATA[15 : 0]

GLOBAL_STORE_B32                                                                      26

Store 32 bits of data from vector input registers into the global aperture.

  MEM[ADDR].b32 = VDATA[31 : 0]

GLOBAL_STORE_B64                                                                      27

Store 64 bits of data from vector input registers into the global aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32]

GLOBAL_STORE_B96                                                                      28

Store 96 bits of data from vector input registers into the global aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32];
  MEM[ADDR + 8U].b32 = VDATA[95 : 64]

GLOBAL_STORE_B128                                                                     29

Store 128 bits of data from vector input registers into the global aperture.

  MEM[ADDR].b32 = VDATA[31 : 0];
  MEM[ADDR + 4U].b32 = VDATA[63 : 32];

  MEM[ADDR + 8U].b32 = VDATA[95 : 64];
  MEM[ADDR + 12U].b32 = VDATA[127 : 96]

GLOBAL_LOAD_D16_U8                                                                                                 30

Load 8 bits of unsigned data from the global aperture, zero extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_I8                                                                                                 31

Load 8 bits of signed data from the global aperture, sign extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

  VDATA[15 : 0].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_B16                                                                                                32

Load 16 bits of unsigned data from the global aperture and store the result into the low 16 bits of a 32-bit vector
register.

  VDATA[15 : 0].b16 = MEM[ADDR].b16;
  // VDATA[31:16] is preserved.

GLOBAL_LOAD_D16_HI_U8                                                                                              33

Load 8 bits of unsigned data from the global aperture, zero extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // VDATA[15:0] is preserved.

GLOBAL_LOAD_D16_HI_I8                                                                                              34

Load 8 bits of signed data from the global aperture, sign extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

  VDATA[31 : 16].i16 = 16'I(signext(MEM[ADDR].i8));
  // VDATA[15:0] is preserved.

GLOBAL_LOAD_D16_HI_B16                                                                                              35

Load 16 bits of unsigned data from the global aperture and store the result into the high 16 bits of a 32-bit
vector register.

  VDATA[31 : 16].b16 = MEM[ADDR].b16;
  // VDATA[15:0] is preserved.

GLOBAL_STORE_D16_HI_B8                                                                                              36

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the global aperture.

  MEM[ADDR].b8 = VDATA[23 : 16]

GLOBAL_STORE_D16_HI_B16                                                                                             37

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the global aperture.

  MEM[ADDR].b16 = VDATA[31 : 16]

GLOBAL_LOAD_ADDTID_B32                                                                                              40

Load 32 bits of data from the global aperture into a vector register. The memory base address is provided in a
scalar register and the lane ID is used as an offset.

  RETURN_DATA.u32 = MEM[SGPR_ADDR[63 : 0] + INST_OFFSET[11 : 0].b64 + 64'B(laneID.i32 * 4)].u32

GLOBAL_STORE_ADDTID_B32                                                                                             41

Store 32 bits of data from a vector input register into the global aperture. The memory base address is provided

as an immediate value and the lane ID is used as an offset.

  MEM[SGPR_ADDR[63 : 0] + INST_OFFSET[11 : 0].b64 + 64'B(laneID.i32 * 4)].u32 = DATA.u32

GLOBAL_ATOMIC_SWAP_B32                                                                                            51

Swap an unsigned 32-bit integer value in the data register with a location in the global aperture. Store the
original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp

GLOBAL_ATOMIC_CMPSWAP_B32                                                                                         52

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in the global
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[ADDR].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_ADD_U32                                                                                             53

Add two unsigned 32-bit integer values stored in the data register and a location in the global aperture. Store
the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  MEM[ADDR].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_SUB_U32                                                                                             54

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  MEM[ADDR].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_CSUB_U32                                                                                              55

Subtract an unsigned 32-bit integer location in the global aperture from a value in the data register and clamp
the result to zero. Store the original value from global aperture into a vector register iff the GLC bit is set.

  declare new_value : 32'U;
  old_value = MEM[ADDR].u32;
  if old_value < DATA.u32 then
      new_value = 0U
  else
      new_value = old_value - DATA.u32
  endif;
  MEM[ADDR].u32 = new_value;
  RETURN_DATA.u32 = old_value

GLOBAL_ATOMIC_MIN_I32                                                                                               56

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].i32;
  src = DATA.i32;
  MEM[ADDR].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

GLOBAL_ATOMIC_MIN_U32                                                                                               57

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_MAX_I32                                                                                               58

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].i32;
  src = DATA.i32;
  MEM[ADDR].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

GLOBAL_ATOMIC_MAX_U32                                                                                               59

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_AND_B32                                                                                               60

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp

GLOBAL_ATOMIC_OR_B32                                                                                                61

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

GLOBAL_ATOMIC_XOR_B32                                                                                             62

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

GLOBAL_ATOMIC_INC_U32                                                                                             63

Increment an unsigned 32-bit integer value from a location in the global aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from global aperture into a vector register iff
the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_DEC_U32                                                                                             64

Decrement an unsigned 32-bit integer value from a location in the global aperture with wraparound to a value
in the data register if the decrement yields a negative value. Store the original value from global aperture into a
vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

GLOBAL_ATOMIC_SWAP_B64                                                                                            65

Swap an unsigned 64-bit integer value in the data register with a location in the global aperture. Store the
original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_CMPSWAP_B64                                                                                           66

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in the global
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[ADDR].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_ADD_U64                                                                                               67

Add two unsigned 64-bit integer values stored in the data register and a location in the global aperture. Store
the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_SUB_U64                                                                                               68

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_MIN_I64                                                                                               69

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

GLOBAL_ATOMIC_MIN_U64                                                                                               70

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_MAX_I64                                                                                               71

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

GLOBAL_ATOMIC_MAX_U64                                                                                               72

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_AND_B64                                                                                               73

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_OR_B64                                                                                              74

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_XOR_B64                                                                                             75

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

GLOBAL_ATOMIC_INC_U64                                                                                             76

Increment an unsigned 64-bit integer value from a location in the global aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from global aperture into a vector register iff
the GLC bit is set.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_DEC_U64                                                                                             77

Decrement an unsigned 64-bit integer value from a location in the global aperture with wraparound to a value
in the data register if the decrement yields a negative value. Store the original value from global aperture into a
vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

GLOBAL_ATOMIC_CMPSWAP_F32                                                                                           80

Compare two single-precision float values stored in the data comparison register and a location in the global
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].f32;
  src = DATA[31 : 0].f32;
  cmp = DATA[63 : 32].f32;
  MEM[ADDR].f32 = tmp == cmp ? src : tmp;
  RETURN_DATA.f32 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_MIN_F32                                                                                               81

Select the minimum of two single-precision float inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  MEM[ADDR].f32 = src < tmp ? src : tmp;
  RETURN_DATA.f32 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_MAX_F32                                                                                               82

Select the maximum of two single-precision float inputs, given two values stored in the data register and a
location in the global aperture. Store the original value from global aperture into a vector register iff the GLC
bit is set.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  MEM[ADDR].f32 = src > tmp ? src : tmp;
  RETURN_DATA.f32 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

GLOBAL_ATOMIC_ADD_F32                                                                                            86

Add two single-precision float values stored in the data register and a location in the global aperture. Store the
original value from global aperture into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].f32;
  MEM[ADDR].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.
