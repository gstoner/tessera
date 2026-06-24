# 16.15. VDS Instructions

> RDNA4 ISA — pages 585–625

16.15. VDS Instructions
This suite of instructions operates on data stored within the data share memory. The instructions transfer data
between VGPRs and data share memory.
The bitfield map for the data-share is:

                 All instructions with RTN in the name return the value that was in memory before the
                operation was performed.

DS_ADD_U32                                                                                                          0

Add two unsigned 32-bit integer values stored in the data register and a location in a data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp

DS_SUB_U32                                                                                                          1

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in a data
share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

DS_RSUB_U32                                                                                                         2

Subtract an unsigned 32-bit integer value stored in a location in a data share from a value stored in the data
register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 = DATA.u32 - MEM[addr].u32;
  RETURN_DATA.u32 = tmp

DS_INC_U32                                                                                                     3

Increment an unsigned 32-bit integer value from a location in a data share with wraparound to 0 if the value
exceeds a value in the data register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

DS_DEC_U32                                                                                                     4

Decrement an unsigned 32-bit integer value from a location in a data share with wraparound to a value in the
data register if the decrement yields a negative value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

DS_MIN_I32                                                                                                     5

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

DS_MAX_I32                                                                                                     6

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;

  RETURN_DATA.i32 = tmp

DS_MIN_U32                                                                                                        7

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

DS_MAX_U32                                                                                                        8

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

DS_AND_B32                                                                                                        9

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in a
data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp

DS_OR_B32                                                                                                        10

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in a data
share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);

  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

DS_XOR_B32                                                                                                       11

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in a
data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

DS_MSKOR_B32                                                                                                     12

Calculate masked bitwise OR on an unsigned 32-bit integer location in a data share, given mask value and bits
to OR in the data registers.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = ((tmp & ~DATA.b32) | DATA2.b32);
  RETURN_DATA.b32 = tmp

DS_STORE_B32                                                                                                     13

Store 32 bits of data from a vector input register into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET.u32].b32 = DATA[31 : 0]

DS_STORE_2ADDR_B32                                                                                               14

Store 32 bits of data from one vector input register and then 32 bits of data from a second vector input register
into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET0.u32 * 4U].b32 = DATA[31 : 0];
  addr = CalcDsAddr(vgpr_a.b32, 0x0);

  MEM[addr + OFFSET1.u32 * 4U].b32 = DATA2[31 : 0]

DS_STORE_2ADDR_STRIDE64_B32                                                                                     15

Store 32 bits of data from one vector input register and then 32 bits of data from a second vector input register
into a data share. Treat each offset as an index and multiply by a stride of 64 elements (256 bytes) to generate
an offset for each DS address.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET0.u32 * 256U].b32 = DATA[31 : 0];
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET1.u32 * 256U].b32 = DATA2[31 : 0]

DS_CMPSTORE_B32                                                                                                 16

Compare an unsigned 32-bit integer value in the data comparison register with a location in a data share, and
modify the memory location with a value in the data source register if the comparison is equal.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  src = DATA.b32;
  cmp = DATA2.b32;
  MEM[addr].b32 = tmp == cmp ? src : tmp;
  RETURN_DATA.b32 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_NUM_F32                                                                                                  18

Select the IEEE minimumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
        MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
        MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
        MEM[ADDR].f32 = src.f32
  elsif ((src.f32 < tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && sign(src.f32) &&
  !sign(tmp.f32))) then

        // NOTE: -0<+0 is TRUE in this comparison
        MEM[ADDR].f32 = src.f32
  else
        MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

DS_MAX_NUM_F32                                                                                              19

Select the IEEE maximumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
        MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
        MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
        MEM[ADDR].f32 = src.f32
  elsif ((src.f32 > tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && !sign(src.f32) &&
  sign(tmp.f32))) then
        // NOTE: +0>-0 is TRUE in this comparison
        MEM[ADDR].f32 = src.f32
  else
        MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

DS_NOP                                                                                                      20

Do nothing.

DS_ADD_F32                                                                                                  21

Add two single-precision float values stored in the data register and a location in a data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].f32;
  MEM[addr].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

DS_STORE_B8                                                                                                      30

Store 8 bits of data from a vector register into a data share.

  MEM[ADDR].b8 = DATA[7 : 0]

DS_STORE_B16                                                                                                     31

Store 16 bits of data from a vector register into a data share.

  MEM[ADDR].b16 = DATA[15 : 0]

DS_ADD_RTN_U32                                                                                                   32

Add two unsigned 32-bit integer values stored in the data register and a location in a data share. Store the
original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp

DS_SUB_RTN_U32                                                                                                   33

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in a data
share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

DS_RSUB_RTN_U32                                                                                                  34

Subtract an unsigned 32-bit integer value stored in a location in a data share from a value stored in the data
register. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 = DATA.u32 - MEM[addr].u32;
  RETURN_DATA.u32 = tmp

DS_INC_RTN_U32                                                                                                 35

Increment an unsigned 32-bit integer value from a location in a data share with wraparound to 0 if the value
exceeds a value in the data register. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

DS_DEC_RTN_U32                                                                                                 36

Decrement an unsigned 32-bit integer value from a location in a data share with wraparound to a value in the
data register if the decrement yields a negative value. Store the original value from data share into a vector
register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

DS_MIN_RTN_I32                                                                                                 37

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

DS_MAX_RTN_I32                                                                                                   38

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

DS_MIN_RTN_U32                                                                                                   39

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

DS_MAX_RTN_U32                                                                                                   40

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

DS_AND_RTN_B32                                                                                                   41

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in a
data share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;

  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp

DS_OR_RTN_B32                                                                                                    42

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in a data
share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

DS_XOR_RTN_B32                                                                                                   43

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in a
data share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

DS_MSKOR_RTN_B32                                                                                                 44

Calculate masked bitwise OR on an unsigned 32-bit integer location in a data share, given mask value and bits
to OR in the data registers.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = ((tmp & ~DATA.b32) | DATA2.b32);
  RETURN_DATA.b32 = tmp

DS_STOREXCHG_RTN_B32                                                                                             45

Swap an unsigned 32-bit integer value in the data register with a location in a data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;

  RETURN_DATA.b32 = tmp

DS_STOREXCHG_2ADDR_RTN_B32                                                                                      46

Swap two unsigned 32-bit integer values in the data registers with two locations in a data share.

  addr1 = ADDR_BASE.u32 + OFFSET0.u32 * 4U;
  addr2 = ADDR_BASE.u32 + OFFSET1.u32 * 4U;
  tmp1 = MEM[addr1].b32;
  tmp2 = MEM[addr2].b32;
  MEM[addr1].b32 = DATA.b32;
  MEM[addr2].b32 = DATA2.b32;
  // Note DATA2 can be any other register
  RETURN_DATA[31 : 0] = tmp1;
  RETURN_DATA[63 : 32] = tmp2

DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32                                                                             47

Swap two unsigned 32-bit integer values in the data registers with two locations in a data share. Treat each
offset as an index and multiply by a stride of 64 elements (256 bytes) to generate an offset for each DS address.

  addr1 = ADDR_BASE.u32 + OFFSET0.u32 * 256U;
  addr2 = ADDR_BASE.u32 + OFFSET1.u32 * 256U;
  tmp1 = MEM[addr1].b32;
  tmp2 = MEM[addr2].b32;
  MEM[addr1].b32 = DATA.b32;
  MEM[addr2].b32 = DATA2.b32;
  // Note DATA2 can be any other register
  RETURN_DATA[31 : 0] = tmp1;
  RETURN_DATA[63 : 32] = tmp2

DS_CMPSTORE_RTN_B32                                                                                             48

Compare an unsigned 32-bit integer value in the data comparison register with a location in a data share, and
modify the memory location with a value in the data source register if the comparison is equal.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b32;
  src = DATA.b32;
  cmp = DATA2.b32;
  MEM[addr].b32 = tmp == cmp ? src : tmp;
  RETURN_DATA.b32 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_NUM_RTN_F32                                                                                                 50

Select the IEEE minimumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value. Store the original value
from data share into a vector register.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
      MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
      MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
      MEM[ADDR].f32 = src.f32
  elsif ((src.f32 < tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && sign(src.f32) &&
  !sign(tmp.f32))) then
      // NOTE: -0<+0 is TRUE in this comparison
      MEM[ADDR].f32 = src.f32
  else
      MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

DS_MAX_NUM_RTN_F32                                                                                                 51

Select the IEEE maximumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value. Store the original value
from data share into a vector register.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
      MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
      MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
      MEM[ADDR].f32 = src.f32
  elsif ((src.f32 > tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && !sign(src.f32) &&
  sign(tmp.f32))) then
      // NOTE: +0>-0 is TRUE in this comparison
      MEM[ADDR].f32 = src.f32
  else
      MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

DS_SWIZZLE_B32                                                                                                      53

Dword swizzle, no data is written to LDS memory.

Swizzles input thread data based on offset mask and returns; note does not read or write the DS memory banks.

Note that reading from an invalid thread results in 0x0.

This opcode supports two specific modes, FFT and rotate, plus two basic modes which swizzle in groups of 4 or
32 consecutive threads.

The FFT mode (offset >= 0xe000) swizzles the input based on offset[4:0] to support FFT calculation. Example
swizzles using input {1, 2, … 20} are:

Offset[4:0]: Swizzle
0x00: {1,11,9,19,5,15,d,1d,3,13,b,1b,7,17,f,1f,2,12,a,1a,6,16,e,1e,4,14,c,1c,8,18,10,20}
0x10: {1,9,5,d,3,b,7,f,2,a,6,e,4,c,8,10,11,19,15,1d,13,1b,17,1f,12,1a,16,1e,14,1c,18,20}
0x1f: No swizzle

The rotate mode (offset >= 0xc000 and offset < 0xe000) rotates the input either left (offset[10] == 0) or right
(offset[10] == 1) a number of threads equal to offset[9:5]. The rotate mode also uses a mask value which can
alter the rotate result. For example, mask == 1 swaps the odd threads across every other even thread (rotate
left), or even threads across every other odd thread (rotate right).

Offset[9:5]: Swizzle
0x01, mask=0, rotate left: {2,3,4,5,6,7,8,9,a,b,c,d,e,f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f,20,1}
0x01, mask=0, rotate right: {20,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f}
0x01, mask=1, rotate left: {1,4,3,6,5,8,7,a,9,c,b,e,d,10,f,12,11,14,13,16,15,18,17,1a,19,1c,1b,1e,1d,20,1f,2}
0x01, mask=1, rotate right: {1f,2,1,4,3,6,5,8,7,a,9,c,b,e,d,10,f,12,11,14,13,16,15,18,17,1a,19,1c,1b,1e,1d,20}

If offset < 0xc000, one of the basic swizzle modes is used based on offset[15]. If offset[15] == 1, groups of 4
consecutive threads are swizzled together. If offset[15] == 0, all 32 threads are swizzled together.

The first basic swizzle mode (when offset[15] == 1) allows full data sharing between a group of 4 consecutive
threads. Any thread within the group of 4 can get data from any other thread within the group of 4, specified by
the corresponding offset bits --- [1:0] for the first thread, [3:2] for the second thread, [5:4] for the third thread,
[7:6] for the fourth thread. Note that the offset bits apply to all groups of 4 within a wavefront; thus if offset[1:0]
== 1, then thread0 grabs thread1, thread4 grabs thread5, etc.

The second basic swizzle mode (when offset[15] == 0) allows limited data sharing between 32 consecutive
threads. In this case, the offset is used to specify a 5-bit xor-mask, 5-bit or-mask, and 5-bit and-mask used to
generate a thread mapping. Note that the offset bits apply to each group of 32 within a wavefront. The details of
the thread mapping are listed below. Some example usages:

SWAPX16 : xor_mask = 0x10, or_mask = 0x00, and_mask = 0x1f

SWAPX8 : xor_mask = 0x08, or_mask = 0x00, and_mask = 0x1f

SWAPX4 : xor_mask = 0x04, or_mask = 0x00, and_mask = 0x1f

SWAPX2 : xor_mask = 0x02, or_mask = 0x00, and_mask = 0x1f

SWAPX1 : xor_mask = 0x01, or_mask = 0x00, and_mask = 0x1f

REVERSEX32 : xor_mask = 0x1f, or_mask = 0x00, and_mask = 0x1f

REVERSEX16 : xor_mask = 0x0f, or_mask = 0x00, and_mask = 0x1f

REVERSEX8 : xor_mask = 0x07, or_mask = 0x00, and_mask = 0x1f

REVERSEX4 : xor_mask = 0x03, or_mask = 0x00, and_mask = 0x1f

REVERSEX2 : xor_mask = 0x01 or_mask = 0x00, and_mask = 0x1f

BCASTX32: xor_mask = 0x00, or_mask = thread, and_mask = 0x00

BCASTX16: xor_mask = 0x00, or_mask = thread, and_mask = 0x10

BCASTX8: xor_mask = 0x00, or_mask = thread, and_mask = 0x18

BCASTX4: xor_mask = 0x00, or_mask = thread, and_mask = 0x1c

BCASTX2: xor_mask = 0x00, or_mask = thread, and_mask = 0x1e

Pseudocode follows:

  offset = offset1:offset0;

  if (offset >= 0xe000) {
      // FFT decomposition
      mask = offset[4:0];
      for (i = 0; i < 64; i++) {
           j = reverse_bits(i & 0x1f);
           j = (j >> count_ones(mask));
           j |= (i & mask);
           j |= i & 0x20;
           thread_out[i] = thread_valid[j] ? thread_in[j] : 0;
      }

  } elsif (offset >= 0xc000) {
      // rotate
      rotate = offset[9:5];
      mask = offset[4:0];
      if (offset[10]) {
           rotate = -rotate;
      }
      for (i = 0; i < 64; i++) {
           j = (i & mask) | ((i + rotate) & ~mask);
           j |= i & 0x20;
           thread_out[i] = thread_valid[j] ? thread_in[j] : 0;
      }

  } elsif (offset[15]) {
      // full data sharing within 4 consecutive threads

      for (i = 0; i < 64; i+=4) {
           thread_out[i+0] = thread_valid[i+offset[1:0]]?thread_in[i+offset[1:0]]:0;
           thread_out[i+1] = thread_valid[i+offset[3:2]]?thread_in[i+offset[3:2]]:0;
           thread_out[i+2] = thread_valid[i+offset[5:4]]?thread_in[i+offset[5:4]]:0;
           thread_out[i+3] = thread_valid[i+offset[7:6]]?thread_in[i+offset[7:6]]:0;
      }

  } else { // offset[15] == 0
      // limited data sharing within 32 consecutive threads
      xor_mask = offset[14:10];
      or_mask = offset[9:5];
      and_mask = offset[4:0];
      for (i = 0; i < 64; i++) {
           j = (((i & 0x1f) & and_mask) | or_mask) ^ xor_mask;
           j |= (i & 0x20); // which group of 32
           thread_out[i] = thread_valid[j] ? thread_in[j] : 0;
      }
  }

DS_LOAD_B32                                                                                                       54

Load 32 bits of data from a data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET.u32].b32

DS_LOAD_2ADDR_B32                                                                                                 55

Load 32 bits of data from one location in a data share and then 32 bits of data from a second location in a data
share and store the results into a 64-bit vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET0.u32 * 4U].b32;
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET1.u32 * 4U].b32

DS_LOAD_2ADDR_STRIDE64_B32                                                                                        56

Load 32 bits of data from one location in a data share and then 32 bits of data from a second location in a data
share and store the results into a 64-bit vector register. Treat each offset as an index and multiply by a stride of
64 elements (256 bytes) to generate an offset for each DS address.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);

  RETURN_DATA[31 : 0] = MEM[addr + OFFSET0.u32 * 256U].b32;
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET1.u32 * 256U].b32

DS_LOAD_I8                                                                                                           57

Load 8 bits of signed data from a data share, sign extend to 32 bits and store the result into a vector register.

  RETURN_DATA.i32 = 32'I(signext(MEM[ADDR].i8))

DS_LOAD_U8                                                                                                           58

Load 8 bits of unsigned data from a data share, zero extend to 32 bits and store the result into a vector register.

  RETURN_DATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })

DS_LOAD_I16                                                                                                          59

Load 16 bits of signed data from a data share, sign extend to 32 bits and store the result into a vector register.

  RETURN_DATA.i32 = 32'I(signext(MEM[ADDR].i16))

DS_LOAD_U16                                                                                                          60

Load 16 bits of unsigned data from a data share, zero extend to 32 bits and store the result into a vector register.

  RETURN_DATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })

DS_CONSUME                                                                                                           61

Subtract (count_bits(exec_mask)) from the value stored in DS memory at instr_offset. Return the pre-operation
value to VGPRs.

The DS returns the pre-op value to all valid lanes. Used for local thread group appends and can be used to
regroup divergent threads.

The compiler uses {offset1,offset0} to provide the relative address to the append counter in the LDS for runtime
index offset or index.

Inside DS, do one atomic add for first valid lane and broadcast result to all valid lanes. Offset = 0ffset1:offset0;
Interpreted as byte offset. Only aligned atomics are supported, so 2 LSBs of offset must be set to zero.

  addr = offset; // offset by LDS HWBASE
  rtnval =   LDS(addr);
  LDS(addr) = LDS(addr) - countbits(valid mask);
  GPR[VDST] = rtnval; // return to all valid threads

DS_APPEND                                                                                                          62

Add (count_bits(exec_mask)) to the value stored in DS memory at instr_offset. Return the pre-operation value
to VGPRs.

The DS returns the pre-op value to all valid lanes. Used for local thread group appends and can be used to
regroup divergent threads.

The compiler uses {offset1,offset0} to provide the relative address to the append counter in the LDS for runtime
index offset or index.

Inside DS, do one atomic add for first valid lane and broadcast result to all valid lanes. Offset = 0ffset1:offset0;
Interpreted as byte offset. Only aligned atomics are supported, so 2 LSBs of offset must be set to zero.

  addr = offset; // offset by LDS HWBASE
  rtnval =   LDS(addr);
  LDS(addr) = LDS(addr) + countbits(valid mask);
  GPR[VDST] = rtnval; // return to all valid threads

DS_ADD_U64                                                                                                         64

Add two unsigned 64-bit integer values stored in the data register and a location in a data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

DS_SUB_U64                                                                                                         65

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in a data
share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

DS_RSUB_U64                                                                                                      66

Subtract an unsigned 64-bit integer value stored in a location in a data share from a value stored in the data
register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 = DATA.u64 - MEM[addr].u64;
  RETURN_DATA.u64 = tmp

DS_INC_U64                                                                                                       67

Increment an unsigned 64-bit integer value from a location in a data share with wraparound to 0 if the value
exceeds a value in the data register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

DS_DEC_U64                                                                                                       68

Decrement an unsigned 64-bit integer value from a location in a data share with wraparound to a value in the
data register if the decrement yields a negative value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

DS_MIN_I64                                                                                                       69

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a

location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MAX_I64                                                                                                     70

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MIN_U64                                                                                                     71

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_MAX_U64                                                                                                     72

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_AND_B64                                                                                                       73

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in a
data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

DS_OR_B64                                                                                                        74

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in a data
share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

DS_XOR_B64                                                                                                       75

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in a
data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

DS_MSKOR_B64                                                                                                     76

Calculate masked bitwise OR on an unsigned 64-bit integer location in a data share, given mask value and bits
to OR in the data registers.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = ((tmp & ~DATA.b64) | DATA2.b64);
  RETURN_DATA.b64 = tmp

DS_STORE_B64                                                                                                    77

Store 64 bits of data from a vector input register into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET.u32].b32 = DATA[31 : 0];
  MEM[addr + OFFSET.u32 + 4U].b32 = DATA[63 : 32]

DS_STORE_2ADDR_B64                                                                                              78

Store 64 bits of data from one vector input register and then 64 bits of data from a second vector input register
into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET0.u32 * 8U].b32 = DATA[31 : 0];
  MEM[addr + OFFSET0.u32 * 8U + 4U].b32 = DATA[63 : 32];
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET1.u32 * 8U].b32 = DATA2[31 : 0];
  MEM[addr + OFFSET1.u32 * 8U + 4U].b32 = DATA2[63 : 32]

DS_STORE_2ADDR_STRIDE64_B64                                                                                     79

Store 64 bits of data from one vector input register and then 64 bits of data from a second vector input register
into a data share. Treat each offset as an index and multiply by a stride of 64 elements (256 bytes) to generate
an offset for each DS address.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET0.u32 * 512U].b32 = DATA[31 : 0];
  MEM[addr + OFFSET0.u32 * 512U + 4U].b32 = DATA[63 : 32];
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET1.u32 * 512U].b32 = DATA2[31 : 0];
  MEM[addr + OFFSET1.u32 * 512U + 4U].b32 = DATA2[63 : 32]

DS_CMPSTORE_B64                                                                                                 80

Compare an unsigned 64-bit integer value in the data comparison register with a location in a data share, and
modify the memory location with a value in the data source register if the comparison is equal.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  src = DATA.b64;
  cmp = DATA2.b64;
  MEM[addr].b64 = tmp == cmp ? src : tmp;

  RETURN_DATA.b64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_NUM_F64                                                                                                82

Select the IEEE minimumNumber() of two double-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  if (isNAN(src.f64) && isNAN(tmp.f64)) then
        MEM[ADDR].f64 = cvtToQuietNAN(src.f64)
  elsif isNAN(src.f64) then
        MEM[ADDR].f64 = tmp.f64
  elsif isNAN(tmp.f64) then
        MEM[ADDR].f64 = src.f64
  elsif ((src.f64 < tmp.f64) || ((abs(src.f64) == 0.0) && (abs(tmp.f64) == 0.0) && sign(src.f64) &&
  !sign(tmp.f64))) then
        // NOTE: -0<+0 is TRUE in this comparison
        MEM[ADDR].f64 = src.f64
  else
        MEM[ADDR].f64 = tmp.f64
  endif;
  RETURN_DATA.f64 = tmp

DS_MAX_NUM_F64                                                                                                83

Select the IEEE maximumNumber() of two double-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  if (isNAN(src.f64) && isNAN(tmp.f64)) then
        MEM[ADDR].f64 = cvtToQuietNAN(src.f64)
  elsif isNAN(src.f64) then
        MEM[ADDR].f64 = tmp.f64
  elsif isNAN(tmp.f64) then
        MEM[ADDR].f64 = src.f64
  elsif ((src.f64 > tmp.f64) || ((abs(src.f64) == 0.0) && (abs(tmp.f64) == 0.0) && !sign(src.f64) &&
  sign(tmp.f64))) then
        // NOTE: +0>-0 is TRUE in this comparison
        MEM[ADDR].f64 = src.f64
  else
        MEM[ADDR].f64 = tmp.f64
  endif;

  RETURN_DATA.f64 = tmp

DS_ADD_RTN_U64                                                                                                   96

Add two unsigned 64-bit integer values stored in the data register and a location in a data share. Store the
original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

DS_SUB_RTN_U64                                                                                                   97

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in a data
share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

DS_RSUB_RTN_U64                                                                                                  98

Subtract an unsigned 64-bit integer value stored in a location in a data share from a value stored in the data
register. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 = DATA.u64 - MEM[addr].u64;
  RETURN_DATA.u64 = tmp

DS_INC_RTN_U64                                                                                                   99

Increment an unsigned 64-bit integer value from a location in a data share with wraparound to 0 if the value
exceeds a value in the data register. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;

  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

DS_DEC_RTN_U64                                                                                                100

Decrement an unsigned 64-bit integer value from a location in a data share with wraparound to a value in the
data register if the decrement yields a negative value. Store the original value from data share into a vector
register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

DS_MIN_RTN_I64                                                                                                101

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MAX_RTN_I64                                                                                                102

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MIN_RTN_U64                                                                                                103

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_MAX_RTN_U64                                                                                                104

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a data share. Update the data share with the selected value. Store the original value from data share
into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_AND_RTN_B64                                                                                                105

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in a
data share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

DS_OR_RTN_B64                                                                                                 106

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in a data
share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

DS_XOR_RTN_B64                                                                                               107

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in a
data share. Store the original value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

DS_MSKOR_RTN_B64                                                                                             108

Calculate masked bitwise OR on an unsigned 64-bit integer location in a data share, given mask value and bits
to OR in the data registers.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = ((tmp & ~DATA.b64) | DATA2.b64);
  RETURN_DATA.b64 = tmp

DS_STOREXCHG_RTN_B64                                                                                         109

Swap an unsigned 64-bit integer value in the data register with a location in a data share.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp

DS_STOREXCHG_2ADDR_RTN_B64                                                                                   110

Swap two unsigned 64-bit integer values in the data registers with two locations in a data share.

  addr1 = ADDR_BASE.u32 + OFFSET0.u32 * 8U;
  addr2 = ADDR_BASE.u32 + OFFSET1.u32 * 8U;
  tmp1 = MEM[addr1].b64;
  tmp2 = MEM[addr2].b64;
  MEM[addr1].b64 = DATA.b64;
  MEM[addr2].b64 = DATA2.b64;
  // Note DATA2 can be any other register
  RETURN_DATA[63 : 0] = tmp1;

  RETURN_DATA[127 : 64] = tmp2

DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64                                                                            111

Swap two unsigned 64-bit integer values in the data registers with two locations in a data share. Treat each
offset as an index and multiply by a stride of 64 elements (256 bytes) to generate an offset for each DS address.

  addr1 = ADDR_BASE.u32 + OFFSET0.u32 * 512U;
  addr2 = ADDR_BASE.u32 + OFFSET1.u32 * 512U;
  tmp1 = MEM[addr1].b64;
  tmp2 = MEM[addr2].b64;
  MEM[addr1].b64 = DATA.b64;
  MEM[addr2].b64 = DATA2.b64;
  // Note DATA2 can be any other register
  RETURN_DATA[63 : 0] = tmp1;
  RETURN_DATA[127 : 64] = tmp2

DS_CMPSTORE_RTN_B64                                                                                            112

Compare an unsigned 64-bit integer value in the data comparison register with a location in a data share, and
modify the memory location with a value in the data source register if the comparison is equal.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].b64;
  src = DATA.b64;
  cmp = DATA2.b64;
  MEM[addr].b64 = tmp == cmp ? src : tmp;
  RETURN_DATA.b64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_NUM_RTN_F64                                                                                             114

Select the IEEE minimumNumber() of two double-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value. Store the original value
from data share into a vector register.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  if (isNAN(src.f64) && isNAN(tmp.f64)) then
        MEM[ADDR].f64 = cvtToQuietNAN(src.f64)
  elsif isNAN(src.f64) then

      MEM[ADDR].f64 = tmp.f64
  elsif isNAN(tmp.f64) then
      MEM[ADDR].f64 = src.f64
  elsif ((src.f64 < tmp.f64) || ((abs(src.f64) == 0.0) && (abs(tmp.f64) == 0.0) && sign(src.f64) &&
  !sign(tmp.f64))) then
      // NOTE: -0<+0 is TRUE in this comparison
      MEM[ADDR].f64 = src.f64
  else
      MEM[ADDR].f64 = tmp.f64
  endif;
  RETURN_DATA.f64 = tmp

DS_MAX_NUM_RTN_F64                                                                                             115

Select the IEEE maximumNumber() of two double-precision float inputs, given two values stored in the data
register and a location in a data share. Update the data share with the selected value. Store the original value
from data share into a vector register.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  if (isNAN(src.f64) && isNAN(tmp.f64)) then
      MEM[ADDR].f64 = cvtToQuietNAN(src.f64)
  elsif isNAN(src.f64) then
      MEM[ADDR].f64 = tmp.f64
  elsif isNAN(tmp.f64) then
      MEM[ADDR].f64 = src.f64
  elsif ((src.f64 > tmp.f64) || ((abs(src.f64) == 0.0) && (abs(tmp.f64) == 0.0) && !sign(src.f64) &&
  sign(tmp.f64))) then
      // NOTE: +0>-0 is TRUE in this comparison
      MEM[ADDR].f64 = src.f64
  else
      MEM[ADDR].f64 = tmp.f64
  endif;
  RETURN_DATA.f64 = tmp

DS_LOAD_B64                                                                                                    118

Load 64 bits of data from a data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET.u32].b32;
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET.u32 + 4U].b32

DS_LOAD_2ADDR_B64                                                                                              119

Load 64 bits of data from one location in a data share and then 64 bits of data from a second location in a data

share and store the results into a 128-bit vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET0.u32 * 8U].b32;
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET0.u32 * 8U + 4U].b32;
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[95 : 64] = MEM[addr + OFFSET1.u32 * 8U].b32;
  RETURN_DATA[127 : 96] = MEM[addr + OFFSET1.u32 * 8U + 4U].b32

DS_LOAD_2ADDR_STRIDE64_B64                                                                                      120

Load 64 bits of data from one location in a data share and then 64 bits of data from a second location in a data
share and store the results into a 128-bit vector register. Treat each offset as an index and multiply by a stride
of 64 elements (256 bytes) to generate an offset for each DS address.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET0.u32 * 512U].b32;
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET0.u32 * 512U + 4U].b32;
  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[95 : 64] = MEM[addr + OFFSET1.u32 * 512U].b32;
  RETURN_DATA[127 : 96] = MEM[addr + OFFSET1.u32 * 512U + 4U].b32

DS_ADD_RTN_F32                                                                                                  121

Add two single-precision float values stored in the data register and a location in a data share. Store the original
value from data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].f32;
  MEM[addr].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

DS_CONDXCHG32_RTN_B64                                                                                           126

Perform 2 conditional write exchanges, where each conditional write exchange writes a 32 bit value from a
data register to a location in data share iff the most significant bit of the data value is set.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;

  declare RETURN_DATA : 32'U[2];
  ADDR = S0.u32;
  DATA = S1.u64;
  offset = { OFFSET1, OFFSET0 };
  ADDR0 = ((ADDR + offset.u32) & 0xfff8U);
  ADDR1 = ADDR0 + 4U;
  RETURN_DATA[0] = LDS[ADDR0].u32;
  if DATA[31] then
      LDS[ADDR0] = { 1'0, DATA[30 : 0] }
  endif;
  RETURN_DATA[1] = LDS[ADDR1].u32;
  if DATA[63] then
      LDS[ADDR1] = { 1'0, DATA[62 : 32] }
  endif

DS_COND_SUB_U32                                                                                                152

Subtract an unsigned 32-bit integer value in the data register from a location in a data share only if the memory
value is greater than or equal to the data register value.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = tmp >= src ? tmp - src : tmp;
  RETURN_DATA.u32 = tmp

DS_SUB_CLAMP_U32                                                                                               153

Subtract an unsigned 32-bit integer location in a data share from a value in the data register and clamp the
result to zero.

  declare new_value : 32'U;
  old_value = MEM[ADDR].u32;
  if old_value < DATA.u32 then
      new_value = 0U
  else
      new_value = old_value - DATA.u32
  endif;
  MEM[ADDR].u32 = new_value;
  RETURN_DATA.u32 = old_value

DS_PK_ADD_F16                                                                                                  154

Add a packed 2-component half-precision float value from the data register to a location in a data share.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].f16 = src[15 : 0].f16 + tmp[15 : 0].f16;
  dst[31 : 16].f16 = src[31 : 16].f16 + tmp[31 : 16].f16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

DS_PK_ADD_BF16                                                                                                  155

Add a packed 2-component BF16 float value from the data register to a location in a data share.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].bf16 = src[15 : 0].bf16 + tmp[15 : 0].bf16;
  dst[31 : 16].bf16 = src[31 : 16].bf16 + tmp[31 : 16].bf16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

DS_STORE_B8_D16_HI                                                                                              160

Store 8 bits of data from the high bits of a vector register into a data share.

  MEM[ADDR].b8 = DATA[23 : 16]

DS_STORE_B16_D16_HI                                                                                             161

Store 16 bits of data from the high bits of a vector register into a data share.

  MEM[ADDR].b16 = DATA[31 : 16]

DS_LOAD_U8_D16                                                                                                  162

Load 8 bits of unsigned data from a data share, zero extend to 16 bits and store the result into the low 16 bits of
a vector register.

  RETURN_DATA[15 : 0].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // RETURN_DATA[31:16] is preserved.

DS_LOAD_U8_D16_HI                                                                                                  163

Load 8 bits of unsigned data from a data share, zero extend to 16 bits and store the result into the high 16 bits of
a vector register.

  RETURN_DATA[31 : 16].u16 = 16'U({ 8'0U, MEM[ADDR].u8 });
  // RETURN_DATA[15:0] is preserved.

DS_LOAD_I8_D16                                                                                                     164

Load 8 bits of signed data from a data share, sign extend to 16 bits and store the result into the low 16 bits of a
vector register.

  RETURN_DATA[15 : 0].i16 = 16'I(signext(MEM[ADDR].i8));
  // RETURN_DATA[31:16] is preserved.

DS_LOAD_I8_D16_HI                                                                                                  165

Load 8 bits of signed data from a data share, sign extend to 16 bits and store the result into the high 16 bits of a
vector register.

  RETURN_DATA[31 : 16].i16 = 16'I(signext(MEM[ADDR].i8));
  // RETURN_DATA[15:0] is preserved.

DS_LOAD_U16_D16                                                                                                    166

Load 16 bits of unsigned data from a data share and store the result into the low 16 bits of a vector register.

  RETURN_DATA[15 : 0].u16 = MEM[ADDR].u16;
  // RETURN_DATA[31:16] is preserved.

DS_LOAD_U16_D16_HI                                                                                                 167

Load 16 bits of unsigned data from a data share and store the result into the high 16 bits of a vector register.

  RETURN_DATA[31 : 16].u16 = MEM[ADDR].u16;
  // RETURN_DATA[15:0] is preserved.

DS_COND_SUB_RTN_U32                                                                                            168

Subtract an unsigned 32-bit integer value in the data register from a location in a data share only if the memory
value is greater than or equal to the data register value. Store the original value from data share into a vector
register.

  addr = CalcDsAddr(vgpr_a.b32, offset.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = tmp >= src ? tmp - src : tmp;
  RETURN_DATA.u32 = tmp

DS_SUB_CLAMP_RTN_U32                                                                                           169

Subtract an unsigned 32-bit integer location in a data share from a value in the data register and clamp the
result to zero. Store the original value from data share into a vector register.

  declare new_value : 32'U;
  old_value = MEM[ADDR].u32;
  if old_value < DATA.u32 then
      new_value = 0U
  else
      new_value = old_value - DATA.u32
  endif;
  MEM[ADDR].u32 = new_value;
  RETURN_DATA.u32 = old_value

DS_PK_ADD_RTN_F16                                                                                              170

Add a packed 2-component half-precision float value from the data register to a location in a data share. Store
the original value from data share into a vector register.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].f16 = src[15 : 0].f16 + tmp[15 : 0].f16;
  dst[31 : 16].f16 = src[31 : 16].f16 + tmp[31 : 16].f16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

DS_PK_ADD_RTN_BF16                                                                                             171

Add a packed 2-component BF16 float value from the data register to a location in a data share. Store the

original value from data share into a vector register.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].bf16 = src[15 : 0].bf16 + tmp[15 : 0].bf16;
  dst[31 : 16].bf16 = src[31 : 16].bf16 + tmp[31 : 16].bf16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

DS_STORE_ADDTID_B32                                                                                             176

Store 32 bits of data from a vector input register into a data share. The memory base address is provided as an
immediate value and the lane ID is used as an offset.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;
  MEM[32'I({ OFFSET1, OFFSET0 } + M0[15 : 0]) + laneID.i32 * 4].u32 = DATA0.u32

DS_LOAD_ADDTID_B32                                                                                              177

Load 32 bits of data from a data share into a vector register. The memory base address is provided as an
immediate value and the lane ID is used as an offset.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;
  RETURN_DATA.u32 = MEM[32'I({ OFFSET1, OFFSET0 } + M0[15 : 0]) + laneID.i32 * 4].u32

DS_PERMUTE_B32                                                                                                  178

Forward permute. This does not access LDS memory and may be called even if no LDS memory is allocated to
the wave. It uses LDS to implement an arbitrary swizzle across threads in a wavefront.

Note the address passed in is the thread ID multiplied by 4.

If multiple sources map to the same destination lane, it is not deterministic which source lane writes to the
destination lane.

See also DS_BPERMUTE_B32.

  // VGPR[laneId][index] is the VGPR RAM
  // VDST, ADDR and DATA0 are from the microcode DS encoding
  declare tmp : 32'B[64];
  declare OFFSET : 16'U;

  declare DATA0 : 32'U;
  declare VDST : 32'U;
  num_lanes = WAVE64 ? 64 : 32;
  for i in 0 : num_lanes - 1 do
        tmp[i] = 0x0
  endfor;
  for i in 0 : num_lanes - 1 do
        // If a source thread is disabled, it does not propagate data.
        if EXEC[i].u1 then
            // ADDR needs to be divided by 4.
            // High-order bits are ignored.
            dst_lane = 32'I(VGPR[i][ADDR] + OFFSET.b32) / 4 % num_lanes;
            tmp[dst_lane] = VGPR[i][DATA0]
        endif
  endfor;
  // Copy data into destination VGPRs. If multiple sources
  // select the same destination thread, the highest-numbered
  // source thread wins.
  for i in 0 : num_lanes - 1 do
        if EXEC[i].u1 then
            VGPR[i][VDST] = tmp[i]
        endif
  endfor

Notes

Examples (simplified 4-thread wavefronts):

        VGPR[SRC0] = { A, B, C, D }
        VGPR[ADDR] = { 0, 0, 12, 4 }
        EXEC = 0xF, OFFSET = 0
        VGPR[VDST] = { B, D, 0, C }

        VGPR[SRC0] = { A, B, C, D }
        VGPR[ADDR] = { 0, 0, 12, 4 }
        EXEC = 0xA, OFFSET = 0
        VGPR[VDST] = { -, D, -, 0 }

DS_BPERMUTE_B32                                                                                             179

Backward permute. This does not access LDS memory and may be called even if no LDS memory is allocated to
the wave. It uses LDS hardware to implement an arbitrary swizzle across threads in a wavefront.

Note the address passed in is the thread ID multiplied by 4.

Note that EXEC mask is applied to both VGPR read and write. If src_lane selects a disabled thread then zero is
returned.

See also DS_PERMUTE_B32.

  // VGPR[laneId][index] is the VGPR RAM
  // VDST, ADDR and DATA0 are from the microcode DS encoding
  declare tmp : 32'B[64];
  declare OFFSET : 16'U;
  declare DATA0 : 32'U;
  declare VDST : 32'U;
  num_lanes = WAVE64 ? 64 : 32;
  for i in 0 : num_lanes - 1 do
        tmp[i] = 0x0
  endfor;
  for i in 0 : num_lanes - 1 do
        // ADDR needs to be divided by 4.
        // High-order bits are ignored.
        src_lane = 32'I(VGPR[i][ADDR] + OFFSET.b32) / 4 % num_lanes;
        // EXEC is applied to the source VGPR reads.
        if EXEC[src_lane].u1 then
            tmp[i] = VGPR[src_lane][DATA0]
        endif
  endfor;
  // Copy data into destination VGPRs. Some source
  // data may be broadcast to multiple lanes.
  for i in 0 : num_lanes - 1 do
        if EXEC[i].u1 then
            VGPR[i][VDST] = tmp[i]
        endif
  endfor

Notes

Examples (simplified 4-thread wavefronts):

        VGPR[SRC0] = { A, B, C, D }
        VGPR[ADDR] = { 0, 0, 12, 4 }
        EXEC = 0xF, OFFSET = 0
        VGPR[VDST] = { A, A, D, B }

        VGPR[SRC0] = { A, B, C, D }
        VGPR[ADDR] = { 0, 0, 12, 4 }
        EXEC = 0xA, OFFSET = 0
        VGPR[VDST] = { -, 0, -, B }

DS_BPERMUTE_FI_B32                                                                                     205

Backward permute and fetch data for invalid lanes. This does not access LDS memory and may be called even
if no LDS memory is allocated to the wave. It uses LDS hardware to implement an arbitrary swizzle across
threads in a wavefront.

Note the address passed in is the thread ID multiplied by 4.

Compare with DS_BPERMUTE_B32.

  // VGPR[laneId][index] is the VGPR RAM
  // VDST, ADDR and DATA0 are from the microcode DS encoding
  declare tmp : 32'B[64];
  declare OFFSET : 16'U;
  declare DATA0 : 32'U;
  declare VDST : 32'U;
  num_lanes = WAVE64 ? 64 : 32;
  for i in 0 : num_lanes - 1 do
      tmp[i] = 0x0
  endfor;
  for i in 0 : num_lanes - 1 do
      // ADDR needs to be divided by 4.
      // High-order bits are ignored.
      src_lane = 32'I(VGPR[i][ADDR] + OFFSET.b32) / 4 % num_lanes;
      // Source VGPR is read even if src_lane is invalid in EXEC mask.
      tmp[i] = VGPR[src_lane][DATA0]
  endfor;
  // Copy data into destination VGPRs. Some source
  // data may be broadcast to multiple lanes.
  for i in 0 : num_lanes - 1 do
      if EXEC[i].u1 then
            VGPR[i][VDST] = tmp[i]
      endif
  endfor

DS_STORE_B96                                                                  222

Store 96 bits of data from a vector input register into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET.u32].b32 = DATA[31 : 0];
  MEM[addr + OFFSET.u32 + 4U].b32 = DATA[63 : 32];
  MEM[addr + OFFSET.u32 + 8U].b32 = DATA[95 : 64]

DS_STORE_B128                                                                 223

Store 128 bits of data from a vector input register into a data share.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  MEM[addr + OFFSET.u32].b32 = DATA[31 : 0];
  MEM[addr + OFFSET.u32 + 4U].b32 = DATA[63 : 32];
  MEM[addr + OFFSET.u32 + 8U].b32 = DATA[95 : 64];
  MEM[addr + OFFSET.u32 + 12U].b32 = DATA[127 : 96]

DS_BVH_STACK_PUSH4_POP1_RTN_B32                                                                                    224

Ray tracing involves traversing a BVH which is a kind of tree where nodes have up to 4 children. Each shader
thread processes one child at a time, and overflow nodes are stored temporarily in LDS using a stack. This
instruction supports pushing/popping the stack to reduce the number of VALU instructions required per
traversal and reduce VMEM bandwidth requirements.

The LDS stack address is computed using values packed into ADDR and part of OFFSET0. ADDR carries the
stack address for the lane. OFFSET0[4:0] contains stack_size[4:0] -- this value is constant for all lanes and is
patched into the shader by software.

OFFSET1[0] carries the triangle size optimization flag. OFFSET1[1] carries the primitive range enable flag.

An updated stack address is returned to ADDR --- note that this VGPR is an in-out operand.

DATA0 contains the last node pointer for BVH.

DATA1 contains up to 4 valid data DWORDs for each thread. At a high level the first 3 DWORDs (DATA1[0:2]) is
pushed to the stack if they are valid, and the last DWORD (DATA1[3]) is returned. If the last DWORD is invalid
then pop the stack and return the value from memory.

  declare stack_base : 32'B;
  declare stack_index : 32'U;
  declare DATA1 : 32'B;
  declare last_node_ptr : 32'B;
  declare INVALID_NODE : 32'B;
  DATA_VALID = lambda(data) (
      if data == INVALID_NODE then
            return 1'0U
      elsif ((last_node_ptr != INVALID_NODE) && (data == last_node_ptr)) then
            // Match last_node_ptr
            return 1'0U
      else
            return 1'1U
      endif);
  // main code
  { stack_base, stack_index } = 64'B(DECODE_ADDR(ADDR, OFFSET0));
  last_node_ptr = DATA0.b32;
  // First 3 passes: push data onto stack
  for i in 0 : 2 do
      if DATA_VALID(DATA1[i * 32 + 31 : i * 32]) then
            MEM[stack_base.u32 + stack_index] = DATA1[i * 32 + 31 : i * 32];
            stack_index += 1U
      elsif DATA1[i].b32 == last_node_ptr then
            // Treat all further data as invalid as well.
            break
      endif
  endfor;
  // Last pass: return data or pop
  if DATA_VALID(DATA1[127 : 96]) then
      RETURN_DATA[31 : 0] = DATA1[127 : 96]
  else
      RETURN_DATA[31 : 0] = MEM[stack_base.u32 + stack_index];
      MEM[stack_base.u32 + stack_index] = INVALID_NODE;
      stack_index -= 1U

  endif;
  RETURN_ADDR[31 : 0] = 32'B(ENCODE_ADDR(stack_base, stack_index))

DS_BVH_STACK_PUSH8_POP1_RTN_B32                                                                                    225

Ray tracing involves traversing a BVH which is a kind of tree where nodes have up to 4 children. Each shader
thread processes one child at a time, and overflow nodes are stored temporarily in LDS using a stack. This
instruction supports pushing/popping the stack to reduce the number of VALU instructions required per
traversal and reduce VMEM bandwidth requirements.

The LDS stack address is computed using values packed into ADDR and part of OFFSET0. ADDR carries the
stack address for the lane. OFFSET0[4:0] contains stack_size[4:0] -- this value is constant for all lanes and is
patched into the shader by software.

OFFSET1[0] carries the triangle size optimization flag. OFFSET1[1] carries the primitive range enable flag.

An updated stack address is returned to ADDR --- note that this VGPR is an in-out operand.

DATA0 contains the last node pointer for BVH.

DATA1 contains up to 8 valid data DWORDs for each thread. At a high level the first 7 DWORDs (DATA1[0:6]) is
pushed to the stack if they are valid, and the last DWORD (DATA1[7]) is returned. If the last DWORD is invalid
then pop the stack and return the value from memory.

  declare stack_base : 32'B;
  declare stack_index : 32'U;
  declare DATA1 : 32'B;
  declare last_node_ptr : 32'B;
  declare INVALID_NODE : 32'B;
  DATA_VALID = lambda(data) (
      if data == INVALID_NODE then
            return 1'0U
      elsif ((last_node_ptr != INVALID_NODE) && (data == last_node_ptr)) then
            // Match last_node_ptr
            return 1'0U
      else
            return 1'1U
      endif);
  // main code
  { stack_base, stack_index } = 64'B(DECODE_ADDR(ADDR, OFFSET0));
  last_node_ptr = DATA0.b32;
  // First 7 passes: push data onto stack
  for i in 0 : 6 do
      if DATA_VALID(DATA1[i * 32 + 31 : i * 32]) then
            MEM[stack_base.u32 + stack_index] = DATA1[i * 32 + 31 : i * 32];
            stack_index += 1U
      elsif DATA1[i].b32 == last_node_ptr then
            // Treat all further data as invalid as well.
            break
      endif
  endfor;
  // Last pass: return data or pop

  if DATA_VALID(DATA1[255 : 224]) then
      RETURN_DATA[31 : 0] = DATA1[255 : 224]
  else
      RETURN_DATA[31 : 0] = MEM[stack_base.u32 + stack_index];
      MEM[stack_base.u32 + stack_index] = INVALID_NODE;
      stack_index -= 1U
  endif;
  RETURN_ADDR[31 : 0] = 32'B(ENCODE_ADDR(stack_base, stack_index))

DS_BVH_STACK_PUSH8_POP2_RTN_B64                                                                                    226

Ray tracing involves traversing a BVH which is a kind of tree where nodes have up to 4 children. Each shader
thread processes one child at a time, and overflow nodes are stored temporarily in LDS using a stack. This
instruction supports pushing/popping the stack to reduce the number of VALU instructions required per
traversal and reduce VMEM bandwidth requirements.

The LDS stack address is computed using values packed into ADDR and part of OFFSET0. ADDR carries the
stack address for the lane. OFFSET0[4:0] contains stack_size[4:0] -- this value is constant for all lanes and is
patched into the shader by software.

OFFSET1[0] carries the triangle size optimization flag. OFFSET1[1] carries the primitive range enable flag.

An updated stack address is returned to ADDR --- note that this VGPR is an in-out operand.

DATA0 contains the last node pointer for BVH.

DATA1 contains up to 8 valid data DWORDs for each thread. At a high level the first 7 DWORDs (DATA1[0:6]) is
pushed to the stack if they are valid, and the last DWORD (DATA1[7]) is returned. If the last DWORD is invalid
then pop the stack twice and return the values from memory.

  declare stack_base : 32'B;
  declare stack_index : 32'U;
  declare DATA1 : 32'B;
  declare last_node_ptr : 32'B;
  declare INVALID_NODE : 32'B;
  DATA_VALID = lambda(data) (
      if data == INVALID_NODE then
           return 1'0U
      elsif ((last_node_ptr != INVALID_NODE) && (data == last_node_ptr)) then
           // Match last_node_ptr
           return 1'0U
      else
           return 1'1U
      endif);
  // main code
  { stack_base, stack_index } = 64'B(DECODE_ADDR(ADDR, OFFSET0));
  last_node_ptr = DATA0.b32;
  // First 7 passes: push data onto stack
  for i in 0 : 6 do
      if DATA_VALID(DATA1[i * 32 + 31 : i * 32]) then
           MEM[stack_base.u32 + stack_index] = DATA1[i * 32 + 31 : i * 32];
           stack_index += 1U

      elsif DATA1[i].b32 == last_node_ptr then
            // Treat all further data as invalid as well.
            break
      endif
  endfor;
  // Last pass: return data or pop
  if DATA_VALID(DATA1[255 : 224]) then
      RETURN_DATA[31 : 0] = DATA1[255 : 224]
  else
      RETURN_DATA[31 : 0] = MEM[stack_base.u32 + stack_index];
      MEM[stack_base.u32 + stack_index] = INVALID_NODE;
      stack_index -= 1U
  endif;
  // Attempt a second pop
  if DATA_VALID(MEM[stack_base.u32 + stack_index]) then
      RETURN_DATA[63 : 32] = MEM[stack_base.u32 + stack_index];
      MEM[stack_base.u32 + stack_index] = INVALID_NODE;
      stack_index -= 1U
  endif;
  RETURN_ADDR[31 : 0] = 32'B(ENCODE_ADDR(stack_base, stack_index))

DS_LOAD_B96                                                               254

Load 96 bits of data from a data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET.u32].b32;
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET.u32 + 4U].b32;
  RETURN_DATA[95 : 64] = MEM[addr + OFFSET.u32 + 8U].b32

DS_LOAD_B128                                                              255

Load 128 bits of data from a data share into a vector register.

  addr = CalcDsAddr(vgpr_a.b32, 0x0);
  RETURN_DATA[31 : 0] = MEM[addr + OFFSET.u32].b32;
  RETURN_DATA[63 : 32] = MEM[addr + OFFSET.u32 + 4U].b32;
  RETURN_DATA[95 : 64] = MEM[addr + OFFSET.u32 + 8U].b32;
  RETURN_DATA[127 : 96] = MEM[addr + OFFSET.u32 + 12U].b32
