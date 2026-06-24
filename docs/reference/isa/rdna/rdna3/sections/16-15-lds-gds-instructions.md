# 16.15. LDS & GDS Instructions

> RDNA3 ISA — pages 511–545

16.15. LDS & GDS Instructions
This suite of instructions operates on data stored within the data share memory. The instructions transfer data
between VGPRs and data share memory.
The bitfield map for the LDS/GDS is:

  OFFSET0   = Unsigned byte offset added to the address from the ADDR VGPR.
  OFFSET1   = Unsigned byte offset added to the address from the ADDR VGPR.
  GDS       = Set if GDS, cleared if LDS.
  OP        = DS instruction opcode
  ADDR      = Source LDS address VGPR 0 - 255.
  DATA0     = Source data0 VGPR 0 - 255.
  DATA1     = Source data1 VGPR 0 - 255.
  VDST      = Destination VGPR 0- 255.

                 All instructions with RTN in the name return the value that was in memory before the
                operation was performed.

DS_ADD_U32                                                                                                      0

Add data register to memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u += DATA.u;
  RETURN_DATA.u = tmp

DS_SUB_U32                                                                                                      1

Subtract data register from memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u -= DATA.u;
  RETURN_DATA.u = tmp

DS_RSUB_U32                                                                                                     2

Subtraction with reversed operands.

  tmp = MEM[ADDR].b;
  MEM[ADDR] = DATA.b - MEM[ADDR].b;
  RETURN_DATA = tmp

DS_INC_U32                                                                                      3

Increment memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u = tmp

DS_DEC_U32                                                                                      4

Decrement memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u = tmp

DS_MIN_I32                                                                                      5

Minimum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src < tmp ? src : tmp;
  RETURN_DATA.i = tmp

DS_MAX_I32                                                                                      6

Maximum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src > tmp ? src : tmp;
  RETURN_DATA.i = tmp

DS_MIN_U32                                                7

Minimum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src < tmp ? src : tmp;
  RETURN_DATA.u = tmp

DS_MAX_U32                                                8

Maximum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src > tmp ? src : tmp;
  RETURN_DATA.u = tmp

DS_AND_B32                                                9

Bitwise AND of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp & DATA.b);
  RETURN_DATA.b = tmp

DS_OR_B32                                                10

Bitwise OR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp | DATA.b);
  RETURN_DATA.b = tmp

DS_XOR_B32                                               11

Bitwise XOR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp ^ DATA.b);
  RETURN_DATA.b = tmp

DS_MSKOR_B32                                                                    12

Masked dword OR, D0 contains the mask and D1 contains the new value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = ((tmp & ~DATA.b) | DATA2.b);
  RETURN_DATA.b = tmp

DS_STORE_B32                                                                    13

Store 32-bit data from a vector register into a given memory location.

  MEM[ADDR] = DATA.b

DS_STORE_2ADDR_B32                                                              14

Write 2 dwords.

  MEM[ADDR_BASE.u + OFFSET0.u * 4U] = DATA.b;
  MEM[ADDR_BASE.u + OFFSET1.u * 4U] = DATA2.b

DS_STORE_2ADDR_STRIDE64_B32                                                     15

Write 2 dwords with larger stride.

  MEM[ADDR_BASE.u + OFFSET0.u * 4U * 64U] = DATA.b;
  MEM[ADDR_BASE.u + OFFSET1.u * 4U * 64U] = DATA2.b

DS_CMPSTORE_B32                                                                 16

Compare and store.

  tmp = MEM[ADDR].b;
  src = DATA.b;
  cmp = DATA2.b;
  MEM[ADDR].b = tmp == cmp ? src : tmp;
  RETURN_DATA.b = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_CMPSTORE_F32                                                                                     17

Floating point compare and store that handles NAN/INF/denormal values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  cmp = DATA2.f;
  MEM[ADDR].f = tmp == cmp ? src : tmp;
  RETURN_DATA.f = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_F32                                                                                          18

Minimum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src < tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_MAX_F32                                                                                          19

Maximum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;

  MEM[ADDR].f = src > tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_NOP                                                     20

Do nothing.

DS_ADD_F32                                                 21

Add data register to floating-point memory value.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src + tmp;
  RETURN_DATA.f = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

DS_STORE_B8                                                30

Byte write.

  MEM[ADDR].b8 = DATA[7 : 0].b8

DS_STORE_B16                                               31

Short write.

  MEM[ADDR].b16 = DATA[15 : 0].b16

DS_ADD_RTN_U32                                             32

Add data register to memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u += DATA.u;
  RETURN_DATA.u = tmp

DS_SUB_RTN_U32                                                                                 33

Subtract data register from memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u -= DATA.u;
  RETURN_DATA.u = tmp

DS_RSUB_RTN_U32                                                                                34

Subtraction with reversed operands.

  tmp = MEM[ADDR].b;
  MEM[ADDR] = DATA.b - MEM[ADDR].b;
  RETURN_DATA = tmp

DS_INC_RTN_U32                                                                                 35

Increment memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u = tmp

DS_DEC_RTN_U32                                                                                 36

Decrement memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u = tmp

DS_MIN_RTN_I32                                   37

Minimum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src < tmp ? src : tmp;
  RETURN_DATA.i = tmp

DS_MAX_RTN_I32                                   38

Maximum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src > tmp ? src : tmp;
  RETURN_DATA.i = tmp

DS_MIN_RTN_U32                                   39

Minimum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src < tmp ? src : tmp;
  RETURN_DATA.u = tmp

DS_MAX_RTN_U32                                   40

Maximum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src > tmp ? src : tmp;
  RETURN_DATA.u = tmp

DS_AND_RTN_B32                                   41

Bitwise AND of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp & DATA.b);
  RETURN_DATA.b = tmp

DS_OR_RTN_B32                                                                 42

Bitwise OR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp | DATA.b);
  RETURN_DATA.b = tmp

DS_XOR_RTN_B32                                                                43

Bitwise XOR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp ^ DATA.b);
  RETURN_DATA.b = tmp

DS_MSKOR_RTN_B32                                                              44

Masked dword OR, D0 contains the mask and D1 contains the new value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = ((tmp & ~DATA.b) | DATA2.b);
  RETURN_DATA.b = tmp

DS_STOREXCHG_RTN_B32                                                          45

Write-exchange operation.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = DATA.b;
  RETURN_DATA.b = tmp

DS_STOREXCHG_2ADDR_RTN_B32                                                                          46

Write-exchange 2 separate dwords.

  addr1 = ADDR_BASE.u + OFFSET0.u * 4U;
  addr2 = ADDR_BASE.u + OFFSET1.u * 4U;
  tmp1 = MEM[addr1].b;
  tmp2 = MEM[addr2].b;
  MEM[addr1].b = DATA.b;
  MEM[addr2].b = DATA2.b;
  // Note DATA2 can be any other register
  RETURN_DATA[31 : 0] = tmp1;
  RETURN_DATA[63 : 32] = tmp2

DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32                                                                 47

Write-exchange 2 separate dwords with a stride of 64 dwords.

  addr1 = ADDR_BASE.u + OFFSET0.u * 4U * 64U;
  addr2 = ADDR_BASE.u + OFFSET1.u * 4U * 64U;
  tmp1 = MEM[addr1].b;
  tmp2 = MEM[addr2].b;
  MEM[addr1].b = DATA.b;
  MEM[addr2].b = DATA2.b;
  // Note DATA2 can be any other register
  RETURN_DATA[31 : 0] = tmp1;
  RETURN_DATA[63 : 32] = tmp2

DS_CMPSTORE_RTN_B32                                                                                 48

Compare and store.

  tmp = MEM[ADDR].b;
  src = DATA.b;
  cmp = DATA2.b;
  MEM[ADDR].b = tmp == cmp ? src : tmp;
  RETURN_DATA.b = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_CMPSTORE_RTN_F32                                                                                 49

Floating point compare and store that handles NAN/INF/denormal values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  cmp = DATA2.f;
  MEM[ADDR].f = tmp == cmp ? src : tmp;
  RETURN_DATA.f = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_RTN_F32                                                                                      50

Minimum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src < tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_MAX_RTN_F32                                                                                      51

Maximum of two floating-point values.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src > tmp ? src : tmp;
  RETURN_DATA = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_WRAP_RTN_B32                                                                                     52

Wrap calculation. Intended for use in ring buffer management.

  tmp = MEM[ADDR].u;

  MEM[ADDR].u = tmp >= DATA.u ? tmp - DATA.u : tmp + DATA2.u;
  RETURN_DATA = tmp

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
0x01, mask=1, rotate left: {2,1,4,7,6,5,8,b,a,9,c,f,e,d,10,13,12,11,14,17,16,15,18,1b,1a,19,1c,1f,1e,1d,20,3}
0x01, mask=1, rotate right: {1e,1,4,3,2,5,8,7,6,9,c,b,a,d,10,f,e,11,14,13,12,15,18,17,16,19,1c,1b,1a,1d,20,1f}

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

DS_LOAD_B32                                                                                   54

Load 32-bit data from a given memory location into a vector register.

  RETURN_DATA = MEM[ADDR].b

DS_LOAD_2ADDR_B32                                                                             55

Read 2 dwords.

  RETURN_DATA[31 : 0] = MEM[ADDR_BASE.u + OFFSET0.u * 4U].b;
  RETURN_DATA[63 : 32] = MEM[ADDR_BASE.u + OFFSET1.u * 4U].b

DS_LOAD_2ADDR_STRIDE64_B32                                                                    56

Read 2 dwords with a larger stride.

  RETURN_DATA[31 : 0] = MEM[ADDR_BASE.u + OFFSET0.u * 4U * 64U].b;

  RETURN_DATA[63 : 32] = MEM[ADDR_BASE.u + OFFSET1.u * 4U * 64U].b

DS_LOAD_I8                                                                                                    57

Signed byte read.

  RETURN_DATA.i = 32'I(signext(MEM[ADDR][7 : 0].i8))

DS_LOAD_U8                                                                                                    58

Unsigned byte read.

  RETURN_DATA.u = 32'U({ 24'0, MEM[ADDR][7 : 0].u8 })

DS_LOAD_I16                                                                                                   59

Signed short read.

  RETURN_DATA.i = 32'I(signext(MEM[ADDR][15 : 0].i16))

DS_LOAD_U16                                                                                                   60

Unsigned short read.

  RETURN_DATA.u = 32'U({ 16'0, MEM[ADDR][15 : 0].u16 })

DS_CONSUME                                                                                                    61

LDS & GDS. Subtract (count_bits(exec_mask)) from the value stored in DS memory at (M0.base + instr_offset).
Return the pre-operation value to VGPRs.

The DS subtracts count_bits(vector valid mask) from the value stored at address M0.base + instruction based
offset and returns the pre-op value to all valid lanes. This op can be used in both the LDS and GDS. In the LDS
this address is an offset to HWBASE and clamped by M0.size, but in the GDS the M0.base constant has the
physical GDS address and the compiler must force offset to zero. In GDS it is for the traditional append buffer
operations. In LDS it is for local thread group appends and can be used to regroup divergent threads. The use

of the M0 register enables the compiler to do indexing of UAV append/consume counters.

For GDS (system wide) consume, the compiler must use a zero for {offset1,offset0}, for LDS the compiler uses
{offset1,offset0} to provide the relative address to the append counter in the LDS for runtime index offset or
index.

Inside DS, do one atomic add for first valid lane and broadcast result to all valid lanes. Offset = 0ffset1:offset0;
Interpreted as byte offset. Only aligned atomics are supported, so 2 lsbs of offset must be set to zero.

  addr = M0.base + offset; // offset by LDS HWBASE, limit to M.size
  rtnval =   LDS(addr);
  LDS(addr) = LDS(addr) - countbits(valid mask);
  GPR[VDST] = rtnval; // return to all valid threads

DS_APPEND                                                                                                          62

LDS & GDS. Add (count_bits(exec_mask)) to the value stored in DS memory at (M0.base + instr_offset). Return
the pre-operation value to VGPRs.

The DS adds count_bits(vector valid mask) from the value stored at address M0.base + instruction based offset
and return the pre-op value to all valid lanes. This op can be used in both the LDS and GDS. In the LDS this
address is an offset to HWBASE and clamped by M0.size, but in the GDS the M0.base constant has the physical
GDS address and the compiler must set offset to zero. In GDS it is for the traditional append buffer operations.
In LDS it is for local thread group appends and can be used to regroup divergent threads. The use of the M0
register enables the compiler to do indexing of UAV append/consume counters.

For GDS (system wide) consume, the compiler must use a zero for {offset1,offset0}, for LDS the compiler uses
{offset1,offset0} to provide the relative address to the append counter in the LDS for runtime index offset or
index.

Inside DS, do one atomic add for first valid lane and broadcast result to all valid lanes. Offset = 0ffset1:offset0;
Interpreted as byte offset. Only aligned atomics are supported, so 2 lsbs of offset must be set to zero.

  addr = M0.base + offset; // offset by LDS HWBASE, limit to M.size
  rtnval =   LDS(addr);
  LDS(addr) = LDS(addr) + countbits(valid mask);
  GPR[VDST] = rtnval; // return to all valid threads

DS_ORDERED_COUNT                                                                                                   63

GDS-only: Intercepted by GDS and processed by ordered append module. The ordered append module queues
request until this request wave is the oldest in the queue at which time the oldest wave request is dispatched to
the DS with an atomic opcode indicated by OFFSET1[5:4].

Unlike append/consume this operation is sent even if there are no valid lanes when it is issued. The GDS adds
zero and advances the tracking walker that needs to match up with the dispatch counter.

The following attributes are encoded in the instruction:

  • OFFSET0[7:2] contains the ordered_count_index (in dwords).
  • OFFSET1[0] contains the wave_release flag.
  • OFFSET1[1] contains the wave_done flag.
  • OFFSET1[5:4] contains the ord_idx_opcode: 2'b00 = DS_ADD_RTN_U32, 2'b01 = DS_STOREXCHG_RTN_B32,
    2'b11 = DS_WRAP_RTN_B32.
  • VGPR_DST is the VGPR the result is written to.
  • VGPR_ADDR specifies the increment in the first valid lane. If no lanes are valid (EXEC = 0) then the
    increment is zero.
  • M0 normally carries {16'gds_base, 16'gds_size} for GDS usage. gds_base[15:2] is ordered_count_base[13:0]
    (in dwords) and gds_size is used to hold the logical_wave_id, the width is based on total number of waves
    in the chip.

The wave type is determined automatically based on the ME_ID and QUEUE_ID of the wavefront.

DS_ADD_U64                                                                                                    64

Add data register to 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

DS_SUB_U64                                                                                                    65

Subtract data register from 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

DS_RSUB_U64                                                                                                   66

Subtraction with reversed operands.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = DATA.b64 - tmp;
  RETURN_DATA.b64 = tmp

DS_INC_U64                                                                                            67

Increment 64-bit memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

DS_DEC_U64                                                                                            68

Decrement 64-bit memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

DS_MIN_I64                                                                                            69

Minimum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MAX_I64                                                                                            70

Maximum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src > tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MIN_U64                                                                                            71

Minimum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_MAX_U64                                                      72

Maximum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src > tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_AND_B64                                                      73

Bitwise AND of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

DS_OR_B64                                                       74

Bitwise OR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

DS_XOR_B64                                                      75

Bitwise XOR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

DS_MSKOR_B64                                                                  76

Masked dword OR, D0 contains the mask and D1 contains the new value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = ((tmp & ~DATA.b64) | DATA2.b64);
  RETURN_DATA.b64 = tmp

DS_STORE_B64                                                                  77

Write qword.

  MEM[ADDR].b64 = DATA.b64

DS_STORE_2ADDR_B64                                                            78

Write 2 qwords.

  MEM[ADDR_BASE.u + OFFSET0.u * 8U].b64 = DATA.b64;
  MEM[ADDR_BASE.u + OFFSET1.u * 8U].b64 = DATA2.b64

DS_STORE_2ADDR_STRIDE64_B64                                                   79

Write 2 qwords with a larger stride.

  MEM[ADDR_BASE.u + OFFSET0.u * 8U * 64U].b64 = DATA.b64;
  MEM[ADDR_BASE.u + OFFSET1.u * 8U * 64U].b64 = DATA2.b64

DS_CMPSTORE_B64                                                               80

Compare and store.

  tmp = MEM[ADDR].b64;
  src = DATA.b64;
  cmp = DATA2.b64;
  MEM[ADDR].b64 = tmp == cmp ? src : tmp;
  RETURN_DATA.b64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_CMPSTORE_F64                                                                                     81

Floating point compare and store that handles NAN/INF/denormal values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  cmp = DATA2.f64;
  MEM[ADDR].f64 = tmp == cmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_F64                                                                                          82

Minimum of two floating-point values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  MEM[ADDR].f64 = src < tmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_MAX_F64                                                                                          83

Maximum of two floating-point values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  MEM[ADDR].f64 = src > tmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_ADD_RTN_U64                                                                                        96

Add data register to 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp

DS_SUB_RTN_U64                                                                                        97

Subtract data register from 64-bit memory value.

  tmp = MEM[ADDR].u64;
  MEM[ADDR].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp

DS_RSUB_RTN_U64                                                                                       98

Subtraction with reversed operands.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = DATA.b64 - tmp;
  RETURN_DATA.b64 = tmp

DS_INC_RTN_U64                                                                                        99

Increment 64-bit memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp

DS_DEC_RTN_U64                                                                                      100

Decrement 64-bit memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u64;

  src = DATA.u64;
  MEM[ADDR].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp

DS_MIN_RTN_I64                                                              101

Minimum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MAX_RTN_I64                                                              102

Maximum of two signed 64-bit integer values.

  tmp = MEM[ADDR].i64;
  src = DATA.i64;
  MEM[ADDR].i64 = src > tmp ? src : tmp;
  RETURN_DATA.i64 = tmp

DS_MIN_RTN_U64                                                              103

Minimum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_MAX_RTN_U64                                                              104

Maximum of two unsigned 64-bit integer values.

  tmp = MEM[ADDR].u64;
  src = DATA.u64;
  MEM[ADDR].u64 = src > tmp ? src : tmp;
  RETURN_DATA.u64 = tmp

DS_AND_RTN_B64                                                              105

Bitwise AND of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp

DS_OR_RTN_B64                                                               106

Bitwise OR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp

DS_XOR_RTN_B64                                                              107

Bitwise XOR of register value and 64-bit memory value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp

DS_MSKOR_RTN_B64                                                            108

Masked dword OR, D0 contains the mask and D1 contains the new value.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = ((tmp & ~DATA.b64) | DATA2.b64);
  RETURN_DATA.b64 = tmp

DS_STOREXCHG_RTN_B64                                                        109

Write-exchange operation.

  tmp = MEM[ADDR].b64;
  MEM[ADDR].b64 = DATA.b64;

  RETURN_DATA.b64 = tmp

DS_STOREXCHG_2ADDR_RTN_B64                                                                        110

Write-exchange 2 separate qwords.

  addr1 = ADDR_BASE.u + OFFSET0.u * 8U;
  addr2 = ADDR_BASE.u + OFFSET1.u * 8U;
  tmp1 = MEM[addr1].b64;
  tmp2 = MEM[addr2].b64;
  MEM[addr1].b64 = DATA.b64;
  MEM[addr2].b64 = DATA2.b64;
  // Note DATA2 can be any other register
  RETURN_DATA[63 : 0] = tmp1;
  RETURN_DATA[127 : 64] = tmp2

DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64                                                               111

Write-exchange 2 qwords with a stride of 64 qwords.

  addr1 = ADDR_BASE.u + OFFSET0.u * 8U * 64U;
  addr2 = ADDR_BASE.u + OFFSET1.u * 8U * 64U;
  tmp1 = MEM[addr1].b64;
  tmp2 = MEM[addr2].b64;
  MEM[addr1].b64 = DATA.b64;
  MEM[addr2].b64 = DATA2.b64;
  // Note DATA2 can be any other register
  RETURN_DATA[63 : 0] = tmp1;
  RETURN_DATA[127 : 64] = tmp2

DS_CMPSTORE_RTN_B64                                                                               112

Compare and store.

  tmp = MEM[ADDR].b64;
  src = DATA.b64;
  cmp = DATA2.b64;
  MEM[ADDR].b64 = tmp == cmp ? src : tmp;
  RETURN_DATA.b64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_CMPSTORE_RTN_F64                                                                               113

Floating point compare and store that handles NAN/INF/denormal values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  cmp = DATA2.f64;
  MEM[ADDR].f64 = tmp == cmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.

DS_MIN_RTN_F64                                                                                    114

Minimum of two floating-point values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  MEM[ADDR].f64 = src < tmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_MAX_RTN_F64                                                                                    115

Maximum of two floating-point values.

  tmp = MEM[ADDR].f64;
  src = DATA.f64;
  MEM[ADDR].f64 = src > tmp ? src : tmp;
  RETURN_DATA.f64 = tmp

Notes

Floating-point compare handles NAN/INF/denorm.

DS_LOAD_B64                                                                                       118

Read 1 qword.

  RETURN_DATA = MEM[ADDR].b64

DS_LOAD_2ADDR_B64                                                                                              119

Read 2 qwords.

  RETURN_DATA[63 : 0] = MEM[ADDR_BASE.u + OFFSET0.u * 8U].b64;
  RETURN_DATA[127 : 64] = MEM[ADDR_BASE.u + OFFSET1.u * 8U].b64

DS_LOAD_2ADDR_STRIDE64_B64                                                                                     120

Read 2 qwords with a larger stride.

  RETURN_DATA[63 : 0] = MEM[ADDR_BASE.u + OFFSET0.u * 8U * 64U].b64;
  RETURN_DATA[127 : 64] = MEM[ADDR_BASE.u + OFFSET1.u * 8U * 64U].b64

DS_ADD_RTN_F32                                                                                                 121

Add data register to floating-point memory value.

  tmp = MEM[ADDR].f;
  src = DATA.f;
  MEM[ADDR].f = src + tmp;
  RETURN_DATA.f = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

DS_ADD_GS_REG_RTN                                                                                              122

Perform an atomic add to data in specific registers embedded in GDS rather than operating on GDS memory
directly. This instruction returns the pre-op value. This instruction is only used by the GS stage and is used to
facilitate streamout.

The return value may be 32 bits or 64 bits depending on the GS register accessed. The data value is 32 bits.

  if OFFSET0[5:2] > 7
      // 64-bit GS register access
      addr = (OFFSET0[5:2] - 8) * 2 + 8;
      VDST[0] = GS_REGS(addr + 0);
      VDST[1] = GS_REGS(addr + 1);
      {GS_REGS(addr + 1), GS_REGS(addr)} += DATA0[0]; // source is 32 bit
  else
      addr = OFFSET0[5:2];
      VDST[0] = GS_REGS(addr);
      GS_REGS(addr) += DATA0[0];
  endif.

32-bit GS registers:

offset[5:2] Register
0 GDS_STRMOUT_BUFFER_FILLED_SIZE_0
1 GDS_STRMOUT_BUFFER_FILLED_SIZE_1
2 GDS_STRMOUT_BUFFER_FILLED_SIZE_2
3 GDS_STRMOUT_BUFFER_FILLED_SIZE_3
4 GDS_GS_0
5 GDS_GS_1
6 GDS_GS_2
7 GDS_GS_3

64-bit GS registers:

offset[5:2] Register
8 GDS_STRMOUT_PRIMS_NEEDED_0
9 GDS_STRMOUT_PRIMS_WRITTEN_0
10 GDS_STRMOUT_PRIMS_NEEDED_1
11 GDS_STRMOUT_PRIMS_WRITTEN_1
12 GDS_STRMOUT_PRIMS_NEEDED_2
13 GDS_STRMOUT_PRIMS_WRITTEN_2
14 GDS_STRMOUT_PRIMS_NEEDED_3
15 GDS_STRMOUT_PRIMS_WRITTEN_3

DS_SUB_GS_REG_RTN                                                                                              123

Perform an atomic subtraction from data in specific registers embedded in GDS rather than operating on GDS
memory directly. This instruction returns the pre-op value. This instruction is only used by the GS stage and is
used to facilitate streamout.

The return value may be 32 bits or 64 bits depending on the GS register accessed. The data value is 32 bits.

  if OFFSET0[5:2] > 7
      // 64-bit GS register access
      addr = (OFFSET0[5:2] - 8) * 2 + 8;
      VDST[0] = GS_REGS(addr + 0);
      VDST[1] = GS_REGS(addr + 1);

      {GS_REGS(addr + 1), GS_REGS(addr)} -= DATA0[0]; // source is 32 bit
  else
      addr = OFFSET0[5:2];
      VDST[0] = GS_REGS(addr);
      GS_REGS(addr) -= DATA0[0];
  endif.

32-bit GS registers:

offset[5:2] Register
0 GDS_STRMOUT_BUFFER_FILLED_SIZE_0
1 GDS_STRMOUT_BUFFER_FILLED_SIZE_1
2 GDS_STRMOUT_BUFFER_FILLED_SIZE_2
3 GDS_STRMOUT_BUFFER_FILLED_SIZE_3
4 GDS_GS_0
5 GDS_GS_1
6 GDS_GS_2
7 GDS_GS_3

64-bit GS registers:

offset[5:2] Register
8 GDS_STRMOUT_PRIMS_NEEDED_0
9 GDS_STRMOUT_PRIMS_WRITTEN_0
10 GDS_STRMOUT_PRIMS_NEEDED_1
11 GDS_STRMOUT_PRIMS_WRITTEN_1
12 GDS_STRMOUT_PRIMS_NEEDED_2
13 GDS_STRMOUT_PRIMS_WRITTEN_2
14 GDS_STRMOUT_PRIMS_NEEDED_3
15 GDS_STRMOUT_PRIMS_WRITTEN_3

DS_CONDXCHG32_RTN_B64                                                            126

Conditional write exchange.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;
  declare RETURN_DATA : 32'U[2];
  ADDR = S0.u;
  DATA = S1.u64;
  offset = { OFFSET1, OFFSET0 };
  ADDR0 = ((ADDR + offset.u) & 0xfff8U);
  ADDR1 = ADDR0 + 4U;
  RETURN_DATA[0] = LDS[ADDR0].u;
  if DATA[31] then
      LDS[ADDR0] = { 1'0, DATA[30 : 0] }
  endif;
  RETURN_DATA[1] = LDS[ADDR1].u;
  if DATA[63] then
      LDS[ADDR1] = { 1'0, DATA[62 : 32] }

  endif

DS_STORE_B8_D16_HI                                                      160

Byte write in to high word.

  MEM[ADDR].b8 = DATA[23 : 16].b8

DS_STORE_B16_D16_HI                                                     161

Short write in to high word.

  MEM[ADDR].b16 = DATA[31 : 16].b16

DS_LOAD_U8_D16                                                          162

Unsigned byte read with masked return to lower word.

  RETURN_DATA[15 : 0].u16 = 16'U({ 8'0U, MEM[ADDR][7 : 0].u8 })

DS_LOAD_U8_D16_HI                                                       163

Unsigned byte read with masked return to upper word.

  RETURN_DATA[31 : 16].u16 = 16'U({ 8'0U, MEM[ADDR][7 : 0].u8 })

DS_LOAD_I8_D16                                                          164

Signed byte read with masked return to lower word.

  RETURN_DATA[15 : 0].i16 = 16'I(signext(MEM[ADDR][7 : 0].i8))

DS_LOAD_I8_D16_HI                                                       165

Signed byte read with masked return to upper word.

  RETURN_DATA[31 : 16].i16 = 16'I(signext(MEM[ADDR][7 : 0].i8))

DS_LOAD_U16_D16                                                                                                    166

Unsigned short read with masked return to lower word.

  RETURN_DATA[15 : 0].u16 = MEM[ADDR][15 : 0].u16

DS_LOAD_U16_D16_HI                                                                                                 167

Unsigned short read with masked return to upper word.

  RETURN_DATA[31 : 16].u16 = MEM[ADDR][15 : 0].u16

DS_BVH_STACK_RTN_B32                                                                                               173

Ray tracing involves traversing a BVH which is a kind of tree where nodes have up to 4 children. Each shader
thread processes one child at a time, and overflow nodes are stored temporarily in LDS using a stack. This
instruction supports pushing/popping the stack to reduce the number of VALU instructions required per
traversal and reduce VMEM bandwidth requirements.

The LDS stack address is computed using values packed into ADDR and part of OFFSET1. ADDR carries the
stack address for the lane. OFFSET1[5:4] contains stack_size[1:0] -- this value is constant for all lanes and is
patched into the shader by software. Valid stack sizes are {8, 16, 32, 64}.

A new stack address is returned to ADDR --- note that this VGPR is an in-out operand.

DATA0 contains the last node pointer for BVH.

DATA1 contains up to 4 valid data DWORDs for each thread. At a high level the first 3 DWORDs (DATA1[0:2]) is
pushed to the stack if they are valid, and the last DWORD (DATA1[3]) is returned. If the last DWORD is invalid
then pop the stack and return the value from memory.

In general this instruction performs the following :

      (stack_base, stack_index) = DECODE_ADDR(ADDR, OFFSET1);
      last_node_ptr = DATA0;
      // First 3 passes: push data onto stack
      for i = 0..2 do
           if DATA_VALID(DATA1[i])

               MEM[stack_base + stack_index] = DATA1[i];
               Increment stack_index
           elsif DATA1[i] == last_node_ptr
               // Treat all further data as invalid as well.
               break
           endif
      endfor
      // Fourth pass: return data or pop
      if DATA_VALID(DATA1[3])
           VGPR_RTN = DATA1[3]
      else
           VGPR_RTN = MEM[stack_base + stack_index];
           MEM[stack_base + stack_index] = INVALID_NODE;
           Decrement stack_index
      endif
      ADDR = ENCODE_ADDR(stack_base, stack_index).

  function DATA_VALID(data):
      if data == INVALID_NODE
           return false
      elsif last_node_ptr != INVALID_NODE && data == last_node_ptr
           // Match last_node_ptr
           return false
      else
           return true
      endif
  endfunction.

DS_STORE_ADDTID_B32                                                                    176

Write dword with thread ID offset.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;
  MEM[32'I({ OFFSET1, OFFSET0 } + M0[15 : 0]) + laneID.i * 4].u = DATA0.u

DS_LOAD_ADDTID_B32                                                                     177

Read dword with thread ID offset.

  declare OFFSET0 : 8'U;
  declare OFFSET1 : 8'U;
  RETURN_DATA.u = MEM[32'I({ OFFSET1, OFFSET0 } + M0[15 : 0]) + laneID.i * 4].u

DS_PERMUTE_B32                                                                         178

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
  for i in 0 : WAVE64 ? 63 : 31 do
        tmp[i] = 0x0
  endfor;
  for i in 0 : WAVE64 ? 63 : 31 do
        // If a source thread is disabled, it does not propagate data.
        if EXEC[i].u1 then
            // ADDR needs to be divided by 4.
            // High-order bits are ignored.
            // NOTE: destination lane is MOD 32 regardless of wave size.
            dst_lane = 32'I(VGPR[i][ADDR] + OFFSET.b) / 4 % 32;
            tmp[dst_lane] = VGPR[i][DATA0]
        endif
  endfor;
  // Copy data into destination VGPRs. If multiple sources
  // select the same destination thread, the highest-numbered
  // source thread wins.
  for i in 0 : WAVE64 ? 63 : 31 do
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
  for i in 0 : WAVE64 ? 63 : 31 do
        tmp[i] = 0x0
  endfor;
  for i in 0 : WAVE64 ? 63 : 31 do
        // ADDR needs to be divided by 4.
        // High-order bits are ignored.
        // NOTE: destination lane is MOD 32 regardless of wave size.
        src_lane = 32'I(VGPR[i][ADDR] + OFFSET.b) / 4 % 32;
        // EXEC is applied to the source VGPR reads.
        if EXEC[src_lane].u1 then
            tmp[i] = VGPR[src_lane][DATA0]
        endif
  endfor;
  // Copy data into destination VGPRs. Some source
  // data may be broadcast to multiple lanes.
  for i in 0 : WAVE64 ? 63 : 31 do
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

DS_STORE_B96                                       222

Tri-dword write.

  MEM[ADDR + 0U].b = DATA[31 : 0];
  MEM[ADDR + 4U].b = DATA[63 : 32];
  MEM[ADDR + 8U].b = DATA[95 : 64]

DS_STORE_B128                                      223

Quad-dword write.

  MEM[ADDR + 0U].b = DATA[31 : 0];
  MEM[ADDR + 4U].b = DATA[63 : 32];
  MEM[ADDR + 8U].b = DATA[95 : 64];
  MEM[ADDR + 12U].b = DATA[127 : 96]

DS_LOAD_B96                                        254

Tri-dword read.

  RETURN_DATA[31 : 0] = MEM[ADDR + 0U].b;
  RETURN_DATA[63 : 32] = MEM[ADDR + 4U].b;
  RETURN_DATA[95 : 64] = MEM[ADDR + 8U].b

DS_LOAD_B128                                       255

Quad-dword read.

  RETURN_DATA[31 : 0] = MEM[ADDR + 0U].b;
  RETURN_DATA[63 : 32] = MEM[ADDR + 4U].b;
  RETURN_DATA[95 : 64] = MEM[ADDR + 8U].b;
  RETURN_DATA[127 : 96] = MEM[ADDR + 12U].b
