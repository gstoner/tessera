// Program execution owns bytes and lifetime, never numerical kernel bodies.
#include "native_program_runtime.h"
#include "ProgramHostTransportPolicy.h"
#include "ProgramCachePolicy.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>
extern "C" int tessera_rocm_image_acquire(const void *, size_t, const char *,
                                        void **, void **, void **, int *);
extern "C" int tessera_rocm_image_release(void *);
namespace {
const pid_t process = getpid();
struct Memref { void *allocated, *aligned; int64_t offset, elements, stride; };
struct Stage {
  void *lease = nullptr;
  hipFunction_t function{};
  std::array<Memref, 7> refs{};
  std::array<int64_t, 8> scalars{};
  std::array<void *, 43> argv{};
  std::array<unsigned, 6> geometry{};
  std::vector<uint32_t> inputs, outputs;
};
struct Program {
  std::string cacheKey;
  uint64_t allocationBytes = 0;
  uint64_t idleLimit = uint64_t(128) << 20;
  int device = 0;
  hipCtx_t context{};
  hipStream_t stream{};
  bool pinned = false;
  void *pinnedInputs = nullptr, *pinnedReadback = nullptr;
  uint64_t pinnedReadbackBytes = 0;
  std::vector<uint64_t> inputOffsets;
  uint32_t arguments = 0;
  std::vector<TesseraRocmProgramBuffer> contract;
  std::vector<void *> buffers;
  std::vector<std::vector<unsigned char>> snapshots;
  std::vector<Stage> stages;
  std::vector<unsigned char> initialized;
  std::vector<unsigned char> readback;
  std::vector<hipEvent_t> events;
  std::vector<hipGraph_t> graphs;
  std::vector<hipGraphExec_t> graphExecutables;
  // One internal frame for every bounded ABI. A frame quantity is
  // coef * M^a * G^b * N^c * H^d, a..d in {0,1}: M is rows, G is K16 groups,
  // N is output columns and H is hidden K16 groups. A variable an ABI does not
  // carry has capacity 0 and active 1: both row ABIs are the G-, N- and H-free
  // case, the shape frame the N- and H-free case.
  int64_t capacityRows = 0, capacityG = 0, capacityN = 0, capacityH = 0;
  std::vector<TesseraRocmProgramShapeStep> frameSteps;
  int64_t activeRows = 0, activeG = 1, activeN = 1, activeH = 1;
  std::vector<uint64_t> unitBytes;   // bytes at M = G = N = H = 1; 0 outside the frame
  std::vector<uint8_t> slotPowers;   // bit 0: M, bit 1: G, bit 2: N, bit 3: H
  uint64_t generation = 0;
  bool poisoned = true, ready = false, output = false;
};
struct State {
  std::mutex mutex;
  std::map<uint64_t, std::unique_ptr<Program>> programs;
  std::vector<std::unique_ptr<Program>> idle;
  uint64_t idleBytes = 0, hits = 0, misses = 0;
  uint64_t next = 1;
};
// A failed completion retains allocations and host snapshots; implicit library
// teardown must not release storage still referenced by an asynchronous copy.
State &state() { static auto *s = new State; return *s; }
bool cacheEnabled() {
  const char *value = std::getenv("TESSERA_ROCM_PROGRAM_CACHE");
  return !value || std::strcmp(value, "0");
}
template<class T> void keyValue(std::string &key, const T &value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}
uint32_t outputCount(const TesseraRocmProgramStep &) { return 1; }
uint32_t outputCount(const TesseraRocmProgramMultiOutputStep &s) {
  return s.output_count;
}
uint32_t outputSlot(const TesseraRocmProgramStep &s, uint32_t) {
  return s.output;
}
uint32_t outputSlot(const TesseraRocmProgramMultiOutputStep &s, uint32_t i) {
  return s.outputs[i];
}
template<class Step>
std::string programKey(const std::string &arch, uint32_t arguments,
                       uint32_t buffers, const TesseraRocmProgramBuffer *contract,
                       uint32_t steps, const Step *plan) {
  // Exact bytes, not caller pointers or a collision-prone hash. Input values
  // are uploaded on every acquire; only immutable image/ABI ownership is reused.
  std::string key = arch;
  keyValue(key, arguments); keyValue(key, buffers); keyValue(key, steps);
  for (uint32_t i = 0; i < buffers; ++i) {
    const auto &b = contract[i];
    keyValue(key,b.bytes); keyValue(key,b.elements);
    keyValue(key,b.first_write); keyValue(key,b.last_read);
    keyValue(key,b.ownership); keyValue(key,b.reserved);
  }
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &p = plan[i];
    keyValue(key,p.image_bytes);
    key.append(static_cast<const char *>(p.image), p.image_bytes);
    size_t length = std::strlen(p.entry); keyValue(key,length); key.append(p.entry,length);
    keyValue(key,p.input_count); keyValue(key,p.scalar_count);
    const uint32_t outputs = outputCount(p);
    keyValue(key,outputs);
    for (uint32_t j=0;j<outputs;++j) keyValue(key,outputSlot(p,j));
    for (unsigned j=0;j<p.input_count;++j) keyValue(key,p.inputs[j]);
    for (unsigned j=0;j<p.scalar_count;++j) keyValue(key,p.scalars[j]);
    for (unsigned j=0;j<6;++j) keyValue(key,p.geometry[j]);
  }
  return key;
}
bool identity(const Program &p) {
  int d = 0; hipCtx_t context{};
  return getpid() == process && hipGetDevice(&d) == hipSuccess &&
         hipCtxGetCurrent(&context) == hipSuccess &&
         d == p.device && context == p.context;
}
int clean(Program &p) {
  if (p.stream && hipStreamSynchronize(p.stream) != hipSuccess) return 7;
  while (!p.graphExecutables.empty()) {
    if (hipGraphExecDestroy(p.graphExecutables.back()) != hipSuccess) return 9;
    p.graphExecutables.pop_back();
  }
  while (!p.graphs.empty()) {
    if (hipGraphDestroy(p.graphs.back()) != hipSuccess) return 9;
    p.graphs.pop_back();
  }
  while (!p.events.empty()) {
    if (hipEventDestroy(p.events.back()) != hipSuccess) return 9;
    p.events.pop_back();
  }
  if (p.pinnedInputs && hipHostFree(p.pinnedInputs) != hipSuccess) return 9;
  p.pinnedInputs = nullptr;
  if (p.pinnedReadback && hipHostFree(p.pinnedReadback) != hipSuccess) return 9;
  p.pinnedReadback = nullptr;
  for (auto &buffer : p.buffers) {
    if (buffer && hipFree(buffer) != hipSuccess) return 9;
    buffer = nullptr;
  }
  for (auto &stage : p.stages) {
    if (stage.lease && tessera_rocm_image_release(stage.lease)) return 8;
    stage.lease = nullptr;
  }
  if (p.stream && hipStreamDestroy(p.stream) != hipSuccess) return 9;
  p.stream = nullptr;
  return 0;
}
// Admission proved every quantity fits at capacity; active <= capacity.
uint64_t monomial(uint32_t powers, int64_t rows, int64_t groups, int64_t columns = 1,
                  int64_t hidden = 1) {
  return ((powers & 1) ? uint64_t(rows) : 1) * ((powers & 2) ? uint64_t(groups) : 1) *
         ((powers & 4) ? uint64_t(columns) : 1) * ((powers & 8) ? uint64_t(hidden) : 1);
}
uint64_t activeBytes(const Program &p, uint32_t slot, int64_t rows, int64_t groups,
                     int64_t columns, int64_t hidden) {
  return !p.unitBytes.empty() && p.unitBytes[slot]
      ? p.unitBytes[slot] * monomial(p.slotPowers[slot], rows, groups, columns, hidden)
      : p.contract[slot].bytes;
}
uint64_t activeBytes(const Program &p, uint32_t slot) {
  return activeBytes(p, slot, p.activeRows, p.activeG, p.activeN, p.activeH);
}
void applyFrame(Program &p, int64_t rows, int64_t groups, int64_t columns, int64_t hidden) {
  if (!p.capacityRows) return;
  p.activeRows = rows; p.activeG = groups; p.activeN = columns; p.activeH = hidden;
  for (const auto &entry : p.frameSteps) {
    auto &stage = p.stages[entry.step];
    uint64_t scale = monomial(entry.powers, rows, groups, columns, hidden);
    stage.scalars[entry.scalar] = int64_t(scale) * entry.scalar_coef;
    if (entry.grid_axis < 3) {
      uint64_t work = scale * uint64_t(entry.grid_coef);
      stage.geometry[entry.grid_axis] =
          unsigned(work / entry.grid_tile + (work % entry.grid_tile != 0));
    }
    for (uint32_t i = 0; i < stage.inputs.size() + stage.outputs.size(); ++i) {
      uint32_t slot = i < stage.inputs.size() ? stage.inputs[i]
          : stage.outputs[i - stage.inputs.size()];
      stage.refs[i].elements = int64_t(activeBytes(p, slot, rows, groups, columns, hidden) /
          (p.contract[slot].bytes / p.contract[slot].elements));
    }
  }
}
template<class Step>
bool validRows(const TesseraRocmProgramRows *rows, int64_t active,
               uint32_t arguments, uint32_t buffers,
               const TesseraRocmProgramBuffer *contract,
               uint32_t steps, const Step *plan, std::vector<uint64_t> &strides,
               std::vector<TesseraRocmProgramRowStep> &rowSteps) {
  strides.assign(buffers, 0);
  rowSteps.clear();
  if (!rows) return active == 0;
  if (rows->capacity_m < 1 || active < 1 || active > rows->capacity_m ||
      rows->slot_count != 3 ||
      rows->consumer_step != steps - 1 || rows->m_scalar != 0 ||
      rows->grid_axis != 1 || !rows->tile_m) return false;
  const auto &consumer = plan[rows->consumer_step];
  // This additive admission is the named A/B/SA/SB/output plus M/N/K
  // consumer ABI. Labels cannot redirect the active M write into N or K.
  if (consumer.input_count != 4 || consumer.scalar_count != 3 ||
      outputCount(consumer) != 1 || rows->slots[0] != consumer.inputs[0] ||
      rows->slots[1] != consumer.inputs[2] ||
      rows->slots[2] != outputSlot(consumer, 0) ||
      consumer.scalars[1] < 1 || consumer.scalars[2] < 1 ||
      consumer.scalars[2] % 16 || rows->row_bytes[0] != uint64_t(consumer.scalars[2]) ||
      rows->row_bytes[1] % 4 ||
      rows->row_bytes[1] / 4 != uint64_t(consumer.scalars[2] / 16)) return false;
  uint64_t grid = uint64_t(rows->capacity_m) / rows->tile_m +
                  (uint64_t(rows->capacity_m) % rows->tile_m != 0);
  if (rows->m_scalar >= consumer.scalar_count ||
      consumer.scalars[rows->m_scalar] != rows->capacity_m ||
      outputCount(consumer) != 1 || grid > 2147483647U ||
      consumer.geometry[rows->grid_axis] != grid) return false;
  uint32_t inputs = 0, outputs = 0;
  for (uint32_t i = 0; i < 7; ++i) {
    if (i >= rows->slot_count) {
      if (rows->slots[i] || rows->row_bytes[i]) return false;
      continue;
    }
    uint32_t slot = rows->slots[i];
    if (slot >= buffers || strides[slot] || !rows->row_bytes[i]) return false;
    const auto &b = contract[slot];
    uint64_t capacity = uint64_t(rows->capacity_m);
    if (b.bytes % capacity || b.elements % capacity ||
        rows->row_bytes[i] != b.bytes / capacity) return false;
    bool consumed = false;
    for (uint32_t j = 0; j < consumer.input_count; ++j)
      consumed |= consumer.inputs[j] == slot;
    if (slot < arguments) {
      if (!consumed) return false;
      ++inputs;
    } else {
      if (b.ownership != 2 || outputSlot(consumer, 0) != slot) return false;
      ++outputs;
    }
    // The invariant producer prefix must not access row-dependent storage.
    for (uint32_t j = 0; j < rows->consumer_step; ++j) {
      for (uint32_t q = 0; q < plan[j].input_count; ++q)
        if (plan[j].inputs[q] == slot) return false;
      for (uint32_t q = 0; q < outputCount(plan[j]); ++q)
        if (outputSlot(plan[j], q) == slot) return false;
    }
    strides[slot] = rows->row_bytes[i];
  }
  const auto &a = contract[rows->slots[0]];
  const auto &sa = contract[rows->slots[1]];
  if (a.bytes / a.elements != 1 || sa.bytes / sa.elements != 4) return false;
  const auto &out = contract[rows->slots[2]];
  uint64_t itemBytes = out.bytes / out.elements;
  if ((itemBytes != 2 && itemBytes != 4) || rows->row_bytes[2] % itemBytes ||
      rows->row_bytes[2] / itemBytes != uint64_t(consumer.scalars[1])) return false;
  if (inputs != 2 || outputs != 1) return false;
  rowSteps.push_back({rows->consumer_step, rows->m_scalar, 1, rows->grid_axis,
                      rows->tile_m, 1});
  return true;
}
// Generic frame admission. Labels are checked against the SSA plan: the row
// closure is derived from slot use, never trusted from the step list alone.
template<class Step>
bool validRowFrame(const TesseraRocmProgramRowFrame *frame, int64_t active,
                   uint32_t arguments, uint32_t buffers,
                   const TesseraRocmProgramBuffer *contract,
                   uint32_t steps, const Step *plan, std::vector<uint64_t> &strides,
                   std::vector<TesseraRocmProgramRowStep> &rowSteps) {
  strides.assign(buffers, 0);
  rowSteps.clear();
  if (!frame || !frame->slots || !frame->row_bytes || !frame->steps ||
      frame->capacity_m < 1 || active < 1 || active > frame->capacity_m ||
      !frame->slot_count || frame->slot_count > 32 || frame->slot_count > buffers ||
      !frame->step_count || frame->step_count > 16 || frame->step_count > steps)
    return false;
  const uint64_t capacity = uint64_t(frame->capacity_m);
  uint32_t rowArguments = 0;
  for (uint32_t i = 0; i < frame->slot_count; ++i) {
    uint32_t slot = frame->slots[i];
    if (slot >= buffers || strides[slot] || !frame->row_bytes[i]) return false;
    const auto &b = contract[slot];
    if (b.bytes % capacity || b.elements % capacity ||
        frame->row_bytes[i] != b.bytes / capacity) return false;
    strides[slot] = frame->row_bytes[i];
    rowArguments += slot < arguments;
  }
  if (!rowArguments) return false;
  std::vector<int> entry(steps, -1);
  for (uint32_t i = 0; i < frame->step_count; ++i) {
    const auto &row = frame->steps[i];
    if (row.step >= steps || entry[row.step] >= 0) return false;
    const auto &s = plan[row.step];
    if (row.scalar >= s.scalar_count || row.scalar_per_row < 1 ||
        frame->capacity_m > INT64_MAX / row.scalar_per_row ||
        s.scalars[row.scalar] != frame->capacity_m * row.scalar_per_row ||
        row.grid_axis > 2 || !row.grid_tile || row.grid_per_row < 1 ||
        capacity > uint64_t(INT64_MAX) / uint64_t(row.grid_per_row)) return false;
    uint64_t work = capacity * uint64_t(row.grid_per_row);
    uint64_t grid = work / row.grid_tile + (work % row.grid_tile != 0);
    if (grid > 2147483647U || s.geometry[row.grid_axis] != grid) return false;
    entry[row.step] = int(i);
  }
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &s = plan[i];
    bool touches = false, invariantOutput = false;
    for (uint32_t j = 0; j < s.input_count; ++j) touches |= strides[s.inputs[j]] != 0;
    for (uint32_t j = 0; j < outputCount(s); ++j) {
      bool row = strides[outputSlot(s, j)] != 0;
      touches |= row; invariantOutput |= !row;
    }
    // A row step may read invariant weights but publishes only row storage.
    if (touches != (entry[i] >= 0) || (touches && invariantOutput)) return false;
  }
  for (uint32_t i = 0; i < frame->step_count; ++i) rowSteps.push_back(frame->steps[i]);
  return true;
}
// Multi-variable frame admission. Every capacity value must equal its
// declared monomial at the capacities; a power on a variable the frame does
// not carry (capacity 0) refuses. The frame closure is derived from SSA slot
// use exactly as for the row frame. One entry scales one scalar (and at most
// one grid axis); a step may carry several entries, never two for the same
// scalar or grid axis.
bool checkedMonomial(uint32_t powers, int64_t coef, const TesseraRocmProgramAxisFrame &f,
                     uint64_t &out) {
  if (coef < 1 || !powers || powers > 15) return false;
  const int64_t extents[4] = {f.capacity_m, f.capacity_g, f.capacity_n, f.capacity_h};
  uint64_t value = uint64_t(coef);
  for (unsigned axis = 0; axis < 4; ++axis) {
    if (!(powers & (1u << axis))) continue;
    if (extents[axis] < 1 || value > uint64_t(INT64_MAX) / uint64_t(extents[axis])) return false;
    value *= uint64_t(extents[axis]);
  }
  out = value;
  return true;
}
template<class Step>
bool validShapeFrame(const TesseraRocmProgramAxisFrame *frame, int64_t rows,
                     int64_t groups, int64_t columns, int64_t hidden,
                     uint32_t arguments, uint32_t buffers,
                     const TesseraRocmProgramBuffer *contract, uint32_t steps,
                     const Step *plan, std::vector<uint64_t> &units,
                     std::vector<uint8_t> &powers,
                     std::vector<TesseraRocmProgramShapeStep> &entries) {
  units.assign(buffers, 0); powers.assign(buffers, 0); entries.clear();
  // M is always framed; G, N and H are framed exactly when their capacity is set.
  auto inRange = [](int64_t active, int64_t capacity) {
    return capacity ? active >= 1 && active <= capacity : active == 1;
  };
  if (!frame || !frame->slots || !frame->unit_bytes || !frame->powers || !frame->steps ||
      frame->capacity_m < 1 || frame->capacity_g < 0 || frame->capacity_n < 0 ||
      frame->capacity_h < 0 ||
      rows < 1 || rows > frame->capacity_m || !inRange(groups, frame->capacity_g) ||
      !inRange(columns, frame->capacity_n) || !inRange(hidden, frame->capacity_h) ||
      !frame->slot_count || frame->slot_count > 32 || frame->slot_count > buffers ||
      !frame->step_count || frame->step_count > 32) return false;
  uint32_t frameArguments = 0;
  for (uint32_t i = 0; i < frame->slot_count; ++i) {
    uint32_t slot = frame->slots[i], power = frame->powers[i];
    uint64_t unit = frame->unit_bytes[i], bytes = 0;
    if (slot >= buffers || units[slot] || !unit || unit > uint64_t(INT64_MAX) ||
        !checkedMonomial(power, int64_t(unit), *frame, bytes))
      return false;
    const auto &b = contract[slot];
    uint64_t item = b.bytes / b.elements;
    if (b.bytes != bytes || unit % item) return false;
    units[slot] = unit; powers[slot] = uint8_t(power);
    frameArguments += slot < arguments;
  }
  if (!frameArguments) return false;
  std::vector<unsigned char> covered(steps, 0);
  for (uint32_t i = 0; i < frame->step_count; ++i) {
    const auto &e = frame->steps[i];
    if (e.step >= steps) return false;
    const auto &s = plan[e.step];
    uint64_t value = 0;
    if (e.scalar >= s.scalar_count ||
        !checkedMonomial(e.powers, e.scalar_coef, *frame, value) ||
        uint64_t(s.scalars[e.scalar]) != value || e.grid_axis > 3) return false;
    if (e.grid_axis < 3) {
      uint64_t work = 0;
      if (!e.grid_tile ||
          !checkedMonomial(e.powers, e.grid_coef, *frame, work))
        return false;
      uint64_t grid = work / e.grid_tile + (work % e.grid_tile != 0);
      if (grid > 2147483647U || s.geometry[e.grid_axis] != grid) return false;
    } else if (e.grid_tile || e.grid_coef) return false;
    for (uint32_t j = 0; j < i; ++j) {
      const auto &o = frame->steps[j];
      if (o.step == e.step && (o.scalar == e.scalar ||
          (e.grid_axis < 3 && o.grid_axis == e.grid_axis))) return false;
    }
    covered[e.step] = 1;
    entries.push_back(e);
  }
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &s = plan[i];
    bool touches = false, invariantOutput = false;
    for (uint32_t j = 0; j < s.input_count; ++j) touches |= units[s.inputs[j]] != 0;
    for (uint32_t j = 0; j < outputCount(s); ++j) {
      bool framed = units[outputSlot(s, j)] != 0;
      touches |= framed; invariantOutput |= !framed;
    }
    // A frame step may read invariant storage but publishes only frame storage.
    if (touches != (covered[i] != 0) || (touches && invariantOutput)) return false;
  }
  return true;
}
bool inputContract(const Program &p, const void *const *inputs,
                   const uint64_t *bytes) {
  if (!inputs || !bytes) return false;
  for (uint32_t i = 0; i < p.arguments; ++i)
    if (!inputs[i] || bytes[i] != activeBytes(p, i)) return false;
  return true;
}
int upload(Program &p, const void *const *inputs, const uint64_t *bytes) {
  if (!inputContract(p, inputs, bytes)) return 1;
  // Completion precedes overwriting either pinned or pageable owned staging.
  // Capacity-sized pageable snapshots are reused across updates and leases.
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  if (!p.pinned) {
    p.snapshots.resize(p.arguments);
    for (uint32_t i = 0; i < p.arguments; ++i)
      p.snapshots[i].resize(size_t(p.contract[i].bytes));
  }
  p.ready = false; p.output = false;
  for (uint32_t i = 0; i < p.arguments; ++i) {
    void *staging = p.pinned
        ? static_cast<unsigned char *>(p.pinnedInputs) + p.inputOffsets[i]
        : p.snapshots[i].data();
    std::memcpy(staging, inputs[i], size_t(bytes[i]));
  }
  for (uint32_t i = 0; i < p.arguments; ++i) {
    const void *source = p.pinned
        ? static_cast<unsigned char *>(p.pinnedInputs) + p.inputOffsets[i]
        : p.snapshots[i].data();
    if (hipMemcpyAsync(p.buffers[i], source, bytes[i],
                       hipMemcpyHostToDevice, p.stream) != hipSuccess) {
      p.poisoned = true; return 5;
    }
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  p.initialized.assign(p.contract.size(), 0);
  std::fill_n(p.initialized.begin(), p.arguments, 1);
  p.ready = true;
  return 0;
}
} // namespace

template<class Step>
static int prepareProgram(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const Step *plan, const void *const *inputs,
    const uint64_t *inputBytes, uint64_t *handle,
    const TesseraRocmProgramRows *rowPolicy = nullptr, int64_t activeRows = 0,
    const TesseraRocmProgramRowFrame *rowFrame = nullptr,
    const TesseraRocmProgramAxisFrame *shapeFrame = nullptr,
    int64_t activeG = 1, int64_t activeN = 1, int64_t activeH = 1) try {
  if (handle) *handle = 0;
  if (!handle || !architecture || !contract || !plan || !inputs || !inputBytes ||
      !arguments || arguments > 128 || !steps || steps > 128 ||
      buffers <= arguments || buffers > 256) return 1;
  if (getpid() != process) return 2;
  std::string arch(architecture);
  if (arch != "gfx1201" && arch != "gfx1151") return 1;
  // Derive lifetimes from the actual SSA edge plan; caller claims must agree.
  std::vector<int64_t> reads(buffers, -1);
  uint64_t total = 0;
  unsigned returned = 0;
  for (uint32_t i = 0; i < buffers; ++i) {
    const auto &b = contract[i];
    if (!b.bytes || !b.elements || b.bytes % b.elements ||
        b.bytes / b.elements > 8 || b.elements > uint64_t(INT64_MAX) ||
        b.reserved || b.ownership > 2 ||
        (i < arguments ? b.ownership != 0 || b.first_write != -1 :
         b.ownership == 0 || b.first_write < 0 || b.first_write >= int64_t(steps)) ||
        b.bytes > (uint64_t(1) << 31) - total) return 1;
    total += b.bytes;
    if (i < arguments && !inputs[i]) return 1;
    if (b.ownership == 2) ++returned;
  }
  if (!returned) return 1;
  uint32_t nextOutput = arguments;
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &s = plan[i];
    const uint32_t outputs = outputCount(s);
    if (!s.image || s.image_bytes < 4 || s.image_bytes > (uint64_t(1) << 28) ||
        std::memcmp(s.image, "\177ELF", 4) || !s.entry || !*s.entry ||
        !s.input_count || s.input_count > 6 || s.scalar_count > 8 ||
        !outputs || outputs > 7 || s.input_count + outputs > 7) return 1;
    for (uint32_t j = 0; j < s.input_count; ++j) {
      if (s.inputs[j] >= nextOutput) return 1;
      reads[s.inputs[j]] = i;
    }
    for (uint32_t j = 0; j < outputs; ++j) {
      if (nextOutput >= buffers || outputSlot(s,j) != nextOutput ||
          contract[nextOutput].first_write != int64_t(i)) return 1;
      reads[nextOutput++] = i;
    }
    uint64_t block = 1;
    for (uint32_t j = 0; j < 6; ++j) {
      if (!s.geometry[j] || s.geometry[j] > 2147483647U) return 1;
      if (j >= 3) {
        if (s.geometry[j] > 1024 || block > 1024 / s.geometry[j]) return 1;
        block *= s.geometry[j];
      }
    }
  }
  if (nextOutput != buffers) return 1;
  for (uint32_t i = 0; i < buffers; ++i) {
    if (contract[i].ownership == 2) reads[i] = steps;
    if (contract[i].last_read != reads[i]) return 1;
  }
  std::vector<uint64_t> units;
  std::vector<uint8_t> powers;
  std::vector<TesseraRocmProgramShapeStep> entries;
  if (int(rowPolicy != nullptr) + int(rowFrame != nullptr) + int(shapeFrame != nullptr) > 1)
    return 1;
  if (shapeFrame) {
    if (!validShapeFrame(shapeFrame, activeRows, activeG, activeN, activeH, arguments, buffers, contract,
                         steps, plan, units, powers, entries)) return 1;
  } else {
    // Both row ABIs are the G-free case of the shape frame.
    std::vector<TesseraRocmProgramRowStep> rowSteps;
    if (activeG != 1 || activeN != 1 || activeH != 1) return 1;
    if (rowFrame ? !validRowFrame(rowFrame, activeRows, arguments, buffers, contract,
                                  steps, plan, units, rowSteps)
                 : !validRows(rowPolicy, activeRows, arguments, buffers, contract,
                              steps, plan, units, rowSteps)) return 1;
    powers.assign(buffers, 0);
    for (uint32_t i = 0; i < buffers; ++i) powers[i] = units[i] ? 1 : 0;
    for (const auto &row : rowSteps)
      entries.push_back({row.step, row.scalar, row.scalar_per_row, 1,
                         row.grid_axis, row.grid_tile, row.grid_per_row});
  }
  for (uint32_t i = 0; i < arguments; ++i)
    if (inputBytes[i] != (units[i] ? units[i] * monomial(powers[i], activeRows, activeG, activeN, activeH)
                                   : contract[i].bytes)) return 1;
  auto p = std::make_unique<Program>();
  if (rowPolicy) p->capacityRows = rowPolicy->capacity_m;
  if (rowFrame) p->capacityRows = rowFrame->capacity_m;
  if (shapeFrame) {
    p->capacityRows = shapeFrame->capacity_m;
    p->capacityG = shapeFrame->capacity_g;
    p->capacityN = shapeFrame->capacity_n;
    p->capacityH = shapeFrame->capacity_h;
  }
  p->frameSteps = std::move(entries);
  p->activeRows = activeRows;
  p->activeG = activeG;
  p->activeN = activeN;
  p->activeH = activeH;
  p->unitBytes = std::move(units);
  p->slotPowers = std::move(powers);
  p->arguments = arguments;
  p->contract.assign(contract, contract + buffers);
  p->buffers.resize(buffers, nullptr);
  p->stages.resize(steps);
  p->events.reserve(2);
  if (hipInit(0) != hipSuccess || hipGetDevice(&p->device) != hipSuccess ||
      hipCtxGetCurrent(&p->context) != hipSuccess) return 2;
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, p->device) != hipSuccess) return 2;
  std::string liveArch(properties.gcnArchName);
  liveArch = liveArch.substr(0, liveArch.find(':'));
  if (liveArch != arch) return 2;
  p->idleLimit = tessera::rocm::programIdleByteBudget(
      arch, uint64_t(properties.totalGlobalMem));
  auto &pool = state();
  std::lock_guard<std::mutex> guard(pool.mutex);
  if (!pool.next) return 12;
  uint64_t inputTotal = 0, outputTotal = 0;
  for (uint32_t i = 0; i < arguments; ++i) {
    p->inputOffsets.push_back(inputTotal);
    inputTotal += contract[i].bytes;
  }
  for (uint32_t i = arguments; i < buffers; ++i)
    if (contract[i].ownership == 2)
      outputTotal += contract[i].bytes;
  // Owning gfx1201 evidence attributes paired-output cost to pageable
  // upload/readback. Select by total traffic while preserving slab bounds,
  // explicit attribution overrides and transport-specific pool identity.
  p->pinned = tessera::rocm::usePinnedProgramHostTransport(
      arch, inputTotal, outputTotal,
      std::getenv("TESSERA_ROCM_PROGRAM_PINNED"));
  p->cacheKey = programKey(arch, arguments, buffers, contract, steps, plan);
  keyValue(p->cacheKey, p->pinned);
  const bool bounded = p->capacityRows != 0;
  keyValue(p->cacheKey, bounded);
  if (bounded) {
    // Every bounded ABI keys the derived frame: capacities, every unit span
    // and its powers, and every scaled launch field. Equal frames share one owner.
    keyValue(p->cacheKey, p->capacityRows); keyValue(p->cacheKey, p->capacityG);
    keyValue(p->cacheKey, p->capacityN); keyValue(p->cacheKey, p->capacityH);
    for (uint32_t i = 0; i < buffers; ++i) {
      keyValue(p->cacheKey, p->unitBytes[i]); keyValue(p->cacheKey, p->slotPowers[i]);
    }
    uint64_t count = p->frameSteps.size();
    keyValue(p->cacheKey, count);
    for (const auto &e : p->frameSteps) {
      keyValue(p->cacheKey, e.step); keyValue(p->cacheKey, e.scalar);
      keyValue(p->cacheKey, e.scalar_coef); keyValue(p->cacheKey, e.powers);
      keyValue(p->cacheKey, e.grid_axis); keyValue(p->cacheKey, e.grid_tile);
      keyValue(p->cacheKey, e.grid_coef);
    }
  }
  p->pinnedReadbackBytes = outputTotal;
  // Budget both pinned slabs and pageable snapshots/readback capacity.
  // Reserved capacity counts even before the first pageable read.
  p->allocationBytes = total + p->cacheKey.size() + inputTotal + outputTotal;
  if (cacheEnabled()) {
    for (auto it = pool.idle.begin(); it != pool.idle.end(); ++it) {
      if ((*it)->poisoned || (*it)->cacheKey != p->cacheKey || !identity(**it)) continue;
      auto reused = std::move(*it);
      pool.idleBytes -= reused->allocationBytes;
      pool.idle.erase(it);
      uint64_t id = pool.next++;
      pool.programs.emplace(id, std::move(reused));
      *handle = id;
      ++pool.hits;
      auto &owned = *pool.programs.at(id);
      // An upload failure quarantines this active owner until explicit close.
      applyFrame(owned, activeRows, activeG, activeN, activeH);
      return upload(owned, inputs, inputBytes);
    }
  }
  ++pool.misses;
  uint64_t id = pool.next++;
  pool.programs.emplace(id, std::move(p));
  *handle = id;
  auto &owned = *pool.programs.at(id);
  if (hipStreamCreateWithFlags(&owned.stream, hipStreamNonBlocking) != hipSuccess) return 4;
  for (uint32_t i = 0; i < steps; ++i) {
    auto &s = owned.stages[i];
    void *module = nullptr, *function = nullptr; int hit = 0;
    if (tessera_rocm_image_acquire(plan[i].image, plan[i].image_bytes,
         plan[i].entry, &s.lease, &module, &function, &hit)) return 3;
    s.function = static_cast<hipFunction_t>(function);
    std::copy_n(plan[i].geometry, 6, s.geometry.begin());
    std::copy_n(plan[i].scalars, plan[i].scalar_count, s.scalars.begin());
  }
  for (uint32_t i = 0; i < buffers; ++i)
    if (hipMalloc(&owned.buffers[i], contract[i].bytes) != hipSuccess) return 4;
  if (owned.pinned) {
    if (hipHostMalloc(&owned.pinnedInputs, inputTotal, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc(&owned.pinnedReadback, outputTotal, hipHostMallocDefault) != hipSuccess)
      return 4;
  }
  for (uint32_t i = 0; i < steps; ++i) {
    auto &s = owned.stages[i]; size_t arg = 0;
    s.inputs.assign(plan[i].inputs, plan[i].inputs + plan[i].input_count);
    s.outputs.clear();
    for (uint32_t j = 0; j < outputCount(plan[i]); ++j)
      s.outputs.push_back(outputSlot(plan[i], j));
    for (uint32_t j = 0; j < plan[i].input_count + outputCount(plan[i]); ++j) {
      uint32_t slot = j < plan[i].input_count ? plan[i].inputs[j] :
          outputSlot(plan[i],j - plan[i].input_count);
      auto &r = s.refs[j];
      r = {owned.buffers[slot], owned.buffers[slot], 0,
           int64_t(contract[slot].elements), 1};
      s.argv[arg++] = &r.allocated; s.argv[arg++] = &r.aligned;
      s.argv[arg++] = &r.offset; s.argv[arg++] = &r.elements; s.argv[arg++] = &r.stride;
    }
    for (uint32_t j = 0; j < plan[i].scalar_count; ++j) s.argv[arg++] = &s.scalars[j];
  }
  applyFrame(owned, activeRows, activeG, activeN, activeH);
  int rc = upload(owned, inputs, inputBytes);
  if (rc) return rc;
  owned.poisoned = false;
  return 0;
} catch (...) { return 12; }


extern "C" int tessera_rocm_program_pack_host_view(
    const void *source, uint64_t sourceSpan, uint32_t rank,
    const uint64_t *shape, const uint64_t *strides, uint32_t itemBytes,
    void *destination, uint64_t destinationBytes) {
  constexpr uint64_t limit = INT64_MAX;
  if (!source || !destination || !shape || !strides || rank < 1 || rank > 32 ||
      !itemBytes || itemBytes > 8 || sourceSpan > limit ||
      destinationBytes > limit) return 1;
  uint64_t count = 1, span = itemBytes;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    if (!shape[axis] || shape[axis] > limit || !strides[axis] ||
        strides[axis] % itemBytes || strides[axis] > limit ||
        count > limit / shape[axis]) return 1;
    count *= shape[axis];
    if (shape[axis] - 1 > (limit - span) / strides[axis]) return 1;
    span += (shape[axis] - 1) * strides[axis];
  }
  if (count > limit / itemBytes || count * itemBytes != destinationBytes ||
      span > sourceSpan) return 1;
  uintptr_t src = reinterpret_cast<uintptr_t>(source);
  uintptr_t dst = reinterpret_cast<uintptr_t>(destination);
  if (src > UINTPTR_MAX - sourceSpan || dst > UINTPTR_MAX - destinationBytes ||
      (src < dst + destinationBytes && dst < src + sourceSpan)) return 1;
  const auto *input = static_cast<const unsigned char *>(source);
  auto *output = static_cast<unsigned char *>(destination);
  // Collapse an actually contiguous suffix; singleton axes contribute no
  // offset. The validated total byte count bounds every block multiplication.
  uint32_t prefix = rank;
  uint64_t blockBytes = itemBytes;
  while (prefix && (shape[prefix-1] == 1 || strides[prefix-1] == blockBytes)) {
    blockBytes *= shape[--prefix];
  }
  if (!prefix) {
    std::memcpy(output, input, destinationBytes);
    return 0;
  }
  const uint64_t columns = shape[prefix-1];
  const uint64_t rows = destinationBytes / blockBytes / columns;
  for (uint64_t row = 0; row < rows; ++row) {
    uint64_t remaining = row, offset = 0;
    for (uint32_t axis = prefix-1; axis-- > 0;) {
      offset += (remaining % shape[axis]) * strides[axis];
      remaining /= shape[axis];
    }
    auto *outRow = output + row * columns * blockBytes;
    const auto *inRow = input + offset;
    // Fixed-width memcpy keeps unaligned byte inputs legal while avoiding a
    // dynamic libc call for each interleaved scalar. Only storage is moved.
    auto fixed = [&](auto width) {
      constexpr size_t bytes = decltype(width)::value;
      for (uint64_t column = 0; column < columns; ++column)
        std::memcpy(outRow + column * bytes,
                    inRow + column * strides[prefix-1], bytes);
    };
    switch (blockBytes) {
    case 1: fixed(std::integral_constant<size_t, 1>{}); break;
    case 2: fixed(std::integral_constant<size_t, 2>{}); break;
    case 4: fixed(std::integral_constant<size_t, 4>{}); break;
    case 8: fixed(std::integral_constant<size_t, 8>{}); break;
    default:
      for (uint64_t column = 0; column < columns; ++column)
        std::memcpy(outRow + column * blockBytes,
                    inRow + column * strides[prefix-1], blockBytes);
    }
  }
  return 0;
}

extern "C" int tessera_rocm_program_prepare(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, uint64_t *handle) {
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle);
}
extern "C" int tessera_rocm_program_prepare_multi_output(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramMultiOutputStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, uint64_t *handle) {
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle);
}

extern "C" int tessera_rocm_program_prepare_multi_output_rows(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramMultiOutputStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, const TesseraRocmProgramRows *rows,
    int64_t activeRows, uint64_t *handle) {
  if (!rows) { if (handle) *handle = 0; return 1; }
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle, rows, activeRows);
}

extern "C" int tessera_rocm_program_prepare_multi_output_row_frame(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramMultiOutputStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, const TesseraRocmProgramRowFrame *frame,
    int64_t activeRows, uint64_t *handle) {
  if (!frame) { if (handle) *handle = 0; return 1; }
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle, nullptr, activeRows, frame);
}

extern "C" int tessera_rocm_program_prepare_multi_output_shape_frame(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramMultiOutputStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, const TesseraRocmProgramShapeFrame *frame,
    int64_t activeRows, int64_t activeG, uint64_t *handle) {
  if (!frame || activeG < 1 || frame->capacity_g < 1) { if (handle) *handle = 0; return 1; }
  // The two-variable shape frame is the N-free axis frame.
  const TesseraRocmProgramAxisFrame axes{frame->capacity_m, frame->capacity_g, 0, 0,
      frame->slot_count, frame->step_count, frame->slots, frame->unit_bytes,
      frame->powers, frame->steps};
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle, nullptr, activeRows,
                        nullptr, &axes, activeG);
}

extern "C" int tessera_rocm_program_prepare_multi_output_axis_frame(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramMultiOutputStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, const TesseraRocmProgramAxisFrame *frame,
    int64_t activeRows, int64_t activeG, int64_t activeN, int64_t activeH,
    uint64_t *handle) {
  if (!frame || activeG < 1 || activeN < 1 || activeH < 1) { if (handle) *handle = 0; return 1; }
  return prepareProgram(architecture, arguments, buffers, contract, steps,
                        plan, inputs, inputBytes, handle, nullptr, activeRows,
                        nullptr, frame, activeG, activeN, activeH);
}

extern "C" int tessera_rocm_program_update(
    uint64_t handle, const void *const *inputs, const uint64_t *bytes) try {
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned) return 10;
  return upload(p, inputs, bytes);
} catch (...) { return 12; }

// Partial root uploads preserve unaffected SSA values, but invalidate every
// transitive dependent result. Complete admission precedes any state change.
static int updateSlots(
    uint64_t handle, uint32_t count, const uint32_t *slots,
    const void *const *inputs, const uint64_t *bytes, int64_t requestedRows,
    int64_t requestedG, int64_t requestedN, int64_t requestedH) try {
  if (!count || !slots || !inputs || !bytes) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.ready) return 10;
  if (count > p.arguments) return 1;
  if (requestedRows && (!p.capacityRows || requestedRows < 1 ||
                        requestedRows > p.capacityRows)) return 1;
  if (requestedG && (!p.capacityG || requestedG < 1 || requestedG > p.capacityG))
    return 1;
  if (requestedN && (!p.capacityN || requestedN < 1 || requestedN > p.capacityN))
    return 1;
  if (requestedH && (!p.capacityH || requestedH < 1 || requestedH > p.capacityH))
    return 1;
  int64_t nextRows = requestedRows ? requestedRows : p.activeRows;
  int64_t nextG = requestedG ? requestedG : p.activeG;
  int64_t nextN = requestedN ? requestedN : p.activeN;
  int64_t nextH = requestedH ? requestedH : p.activeH;
  std::vector<unsigned char> changed(p.contract.size(), 0);
  for (uint32_t i = 0; i < count; ++i) {
    if (slots[i] >= p.arguments || changed[slots[i]] ||
        !inputs[i] || bytes[i] != activeBytes(p, slots[i], nextRows, nextG, nextN, nextH)) return 1;
    changed[slots[i]] = 1;
  }
  // A variable change needs every original slot that depends on it.
  const uint8_t moved = uint8_t((nextRows != p.activeRows ? 1 : 0) |
                                (nextG != p.activeG ? 2 : 0) |
                                (nextN != p.activeN ? 4 : 0) |
                                (nextH != p.activeH ? 8 : 0));
  if (moved)
    for (uint32_t slot = 0; slot < p.arguments; ++slot)
      if (p.unitBytes[slot] && (p.slotPowers[slot] & moved) && !changed[slot]) return 1;
  auto initialized = p.initialized;
  // Private frame slots were produced for the previous active extents; a
  // change invalidates them even where no updated root reaches them.
  if (moved)
    for (uint32_t slot = p.arguments; slot < p.contract.size(); ++slot)
      if (p.unitBytes[slot] && (p.slotPowers[slot] & moved)) {
        changed[slot] = 1; initialized[slot] = 0;
      }
  for (auto &stage : p.stages) {
    bool invalid = false;
    for (auto slot : stage.inputs) invalid |= changed[slot] != 0;
    if (invalid)
      for (auto slot : stage.outputs) {
        changed[slot] = 1;
        initialized[slot] = 0;
      }
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  p.poisoned = true; p.ready = false; p.output = false;
  if (moved) {
    while (!p.graphExecutables.empty()) {
      if (hipGraphExecDestroy(p.graphExecutables.back()) != hipSuccess) return 9;
      p.graphExecutables.pop_back();
    }
    while (!p.graphs.empty()) {
      if (hipGraphDestroy(p.graphs.back()) != hipSuccess) return 9;
      p.graphs.pop_back();
    }
    applyFrame(p, nextRows, nextG, nextN, nextH);
  }
  for (uint32_t i = 0; i < count; ++i) {
    auto slot = slots[i];
    const void *source;
    if (p.pinned) {
      auto *destination = static_cast<unsigned char *>(p.pinnedInputs) + p.inputOffsets[slot];
      std::memcpy(destination, inputs[i], bytes[i]); source = destination;
    } else {
      // Full preparation allocated this slot's capacity-sized owned snapshot.
      // Partial updates preserve that capacity and never replace its storage.
      auto *destination = p.snapshots[slot].data();
      std::memcpy(destination, inputs[i], size_t(bytes[i]));
      source = destination;
    }
    if (hipMemcpyAsync(p.buffers[slot], source, bytes[i],
                      hipMemcpyHostToDevice, p.stream) != hipSuccess) return 5;
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) return 7;
  p.initialized = std::move(initialized);
  p.ready = true; p.poisoned = false;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_update_slots(
    uint64_t handle, uint32_t count, const uint32_t *slots,
    const void *const *inputs, const uint64_t *bytes) {
  return updateSlots(handle, count, slots, inputs, bytes, 0, 0, 0, 0);
}
extern "C" int tessera_rocm_program_update_rows(
    uint64_t handle, int64_t rows, uint32_t count, const uint32_t *slots,
    const void *const *inputs, const uint64_t *bytes) {
  if (rows < 1) return 1;
  return updateSlots(handle, count, slots, inputs, bytes, rows, 0, 0, 0);
}
extern "C" int tessera_rocm_program_update_shape(
    uint64_t handle, int64_t rows, int64_t groups, uint32_t count,
    const uint32_t *slots, const void *const *inputs, const uint64_t *bytes) {
  if (rows < 1 || groups < 1) return 1;
  return updateSlots(handle, count, slots, inputs, bytes, rows, groups, 0, 0);
}
extern "C" int tessera_rocm_program_update_axes(
    uint64_t handle, int64_t rows, int64_t groups, int64_t columns, int64_t hidden,
    uint32_t count, const uint32_t *slots, const void *const *inputs,
    const uint64_t *bytes) {
  if (rows < 1 || groups < 0 || columns < 0 || hidden < 0) return 1;
  // Zero keeps a variable at its active extent (it may be absent).
  return updateSlots(handle, count, slots, inputs, bytes, rows, groups, columns, hidden);
}
extern "C" int tessera_rocm_program_axes_info(
    uint64_t handle, int64_t *capacity, int64_t *active, uint64_t *bytes,
    uint64_t *count) try {
  if (!capacity || !active || !bytes || !count) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (!p.capacityRows) return 1;
  if (p.poisoned) return 10;
  capacity[0] = p.capacityRows; capacity[1] = p.capacityG; capacity[2] = p.capacityN;
  capacity[3] = p.capacityH;
  active[0] = p.activeRows; active[1] = p.activeG; active[2] = p.activeN; active[3] = p.activeH;
  *bytes = p.allocationBytes; *count = p.buffers.size();
  return 0;
} catch (...) { return 12; }
extern "C" int tessera_rocm_program_shape_info(
    uint64_t handle, int64_t *capacityM, int64_t *capacityG, int64_t *activeM,
    int64_t *activeG, uint64_t *bytes, uint64_t *count) try {
  if (!capacityM || !capacityG || !activeM || !activeG || !bytes || !count) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (!p.capacityG) return 1;
  if (p.poisoned) return 10;
  *capacityM = p.capacityRows; *capacityG = p.capacityG;
  *activeM = p.activeRows; *activeG = p.activeG;
  *bytes = p.allocationBytes; *count = p.buffers.size();
  return 0;
} catch (...) { return 12; }
extern "C" int tessera_rocm_program_rows_info(
    uint64_t handle, int64_t *capacity, int64_t *active,
    uint64_t *bytes, uint64_t *count) try {
  if (!capacity || !active || !bytes || !count) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (!p.capacityRows) return 1;
  if (p.poisoned) return 10;
  *capacity = p.capacityRows; *active = p.activeRows;
  *bytes = p.allocationBytes; *count = p.buffers.size();
  return 0;
} catch (...) { return 12; }

static int invokeRange(
    uint64_t handle, uint32_t first, uint32_t count, bool full,
    uint32_t repeats, uint64_t *generation, float *elapsed) try {
  if (generation) *generation = 0;
  if (elapsed) *elapsed = 0;
  if (!generation || !repeats || repeats > 1048576) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.ready || p.generation == UINT64_MAX) return 10;
  if (full) { first = 0; count = p.stages.size(); }
  if (!count || first >= p.stages.size() || count > p.stages.size() - first) return 1;
  auto initialized = p.initialized;
  for (uint32_t i = first; i < first + count; ++i) {
    for (auto slot : p.stages[i].inputs)
      if (!initialized[slot]) return 10;
    for (auto slot : p.stages[i].outputs) initialized[slot] = 1;
  }
  p.output = false;
  hipEvent_t begin{}, end{};
  if (elapsed) {
    if (hipEventCreate(&begin) != hipSuccess) return 4;
    p.events.push_back(begin);
    if (hipEventCreate(&end) != hipSuccess) { p.poisoned = true; return 4; }
    p.events.push_back(end);
    if (hipEventRecord(begin, p.stream) != hipSuccess) { p.poisoned = true; return 6; }
  }
  int status = 0;
  for (uint32_t i = 0; i < repeats && !status; ++i)
    for (uint32_t member = first; member < first + count; ++member) {
      auto &s = p.stages[member];
      if (hipModuleLaunchKernel(s.function, s.geometry[0], s.geometry[1], s.geometry[2],
          s.geometry[3], s.geometry[4], s.geometry[5], 0, p.stream, s.argv.data(),
          nullptr) != hipSuccess) { status = 6; break; }
    }
  if (!status && elapsed && hipEventRecord(end, p.stream) != hipSuccess) status = 6;
  // Completion precedes event destruction, output publication and any readback.
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  if (!status && elapsed && hipEventElapsedTime(elapsed, begin, end) != hipSuccess) status = 7;
  while (!p.events.empty()) {
    if (hipEventDestroy(p.events.back()) != hipSuccess) { p.poisoned = true; return 9; }
    p.events.pop_back();
  }
  if (status) { p.poisoned = true; return status; }
  if (elapsed) *elapsed /= repeats;
  p.initialized = std::move(initialized);
  p.output = true;
  for (uint32_t slot = 0; slot < p.contract.size(); ++slot)
    if (p.contract[slot].ownership == 2 && !p.initialized[slot]) p.output = false;
  *generation = ++p.generation;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_invoke(
    uint64_t handle, uint32_t repeats, uint64_t *generation, float *elapsed) {
  return invokeRange(handle, 0, 0, true, repeats, generation, elapsed);
}
extern "C" int tessera_rocm_program_invoke_range(
    uint64_t handle, uint32_t first, uint32_t count, uint32_t repeats,
    uint64_t *generation, float *elapsed) {
  return invokeRange(handle, first, count, false, repeats, generation, elapsed);
}

// Diagnostic member windows run the immutable SSA members in prefix order.
// Repetition of a pure member writes the same fresh output; consumers only run
// after producer completion. Captured launch groups remove CPU enqueue gaps;
// windows still include device graph dispatch, not instruction-counter time.
extern "C" int tessera_rocm_program_profile_members(
    uint64_t handle, uint32_t repeats, uint32_t members,
    uint64_t *generation, float *elapsed) try {
  if (generation) *generation = 0;
  if (!generation || !elapsed || !repeats || repeats > 65536) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.ready || p.generation == UINT64_MAX) return 10;
  if (members != p.stages.size()) return 1;
  p.graphs.reserve(1); p.graphExecutables.reserve(1);
  p.events.reserve(2);
  p.output = false;
  // Quarantine on any operation failure or exception. Only complete successful
  // profiling publishes a fresh output generation and restores a usable owner.
  p.poisoned = true;
  for (uint32_t member = 0; member < members; ++member) {
    auto &s = p.stages[member];
    if (hipStreamBeginCapture(p.stream, hipStreamCaptureModeThreadLocal) != hipSuccess)
      return 6;
    int status = 0;
    for (uint32_t i = 0; i < repeats; ++i)
      if (hipModuleLaunchKernel(s.function, s.geometry[0], s.geometry[1], s.geometry[2],
          s.geometry[3], s.geometry[4], s.geometry[5], 0, p.stream, s.argv.data(),
          nullptr) != hipSuccess) { status = 6; break; }
    hipGraph_t graph{};
    auto ended = hipStreamEndCapture(p.stream, &graph);
    if (graph) p.graphs.push_back(graph);
    if (status || ended != hipSuccess || !graph) return 6;
    hipGraphExec_t executable{};
    auto instantiated = hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (executable) p.graphExecutables.push_back(executable);
    if (instantiated != hipSuccess || !executable) return 4;
    hipEvent_t begin{}, end{};
    if (hipEventCreate(&begin) != hipSuccess) return 4;
    p.events.push_back(begin);
    if (hipEventCreate(&end) != hipSuccess) return 4;
    p.events.push_back(end);
    if (hipEventRecord(begin, p.stream) != hipSuccess ||
        hipGraphLaunch(executable, p.stream) != hipSuccess ||
        hipEventRecord(end, p.stream) != hipSuccess) return 6;
    if (hipStreamSynchronize(p.stream) != hipSuccess) return 7;
    if (hipEventElapsedTime(&elapsed[member], begin, end) != hipSuccess) return 7;
    elapsed[member] /= repeats;
    // clean first synchronizes, then releases graph/event resources. Here
    // release only these resources; immutable buffers/image leases stay owned.
    if (hipGraphExecDestroy(executable) != hipSuccess) return 9;
    p.graphExecutables.pop_back();
    if (hipGraphDestroy(graph) != hipSuccess) return 9;
    p.graphs.pop_back();
    while (!p.events.empty()) {
      if (hipEventDestroy(p.events.back()) != hipSuccess) return 9;
      p.events.pop_back();
    }
  }
  p.initialized.assign(p.contract.size(), 1);
  p.poisoned = false; p.output = true; *generation = ++p.generation;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_read(
    uint64_t handle, uint32_t slot, uint64_t generation, void *output,
    uint64_t bytes) try {
  if (!output) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (slot >= p.contract.size() || p.contract[slot].ownership != 2 ||
      bytes != activeBytes(p, slot)) return 1;
  if (p.poisoned || !p.output || !generation || generation != p.generation) return 10;
  if (!p.pinned) p.readback.resize(bytes);
  if (p.pinned && bytes > p.pinnedReadbackBytes) return 1;
  void *staging = p.pinned ? p.pinnedReadback : p.readback.data();
  if (hipMemcpyAsync(staging, p.buffers[slot], bytes,
                     hipMemcpyDeviceToHost, p.stream) != hipSuccess) {
    p.poisoned = true; return 5;
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  std::memcpy(output, staging, bytes);
  return 0;
} catch (...) { return 12; }

// Admit the complete destination frame before any enqueue, then publish only
// after one successful completion. Failed copies target retained private
// staging, never borrowed caller output buffers.
extern "C" int tessera_rocm_program_read_many(
    uint64_t handle, uint64_t generation, uint32_t count,
    const uint32_t *slots, void *const *outputs, const uint64_t *bytes) try {
  if (!count || count > 128 || !slots || !outputs || !bytes) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.output || !generation || generation != p.generation) return 10;
  std::array<uint64_t, 128> offsets{};
  uint64_t total = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (!outputs[i] || slots[i] >= p.contract.size() ||
        p.contract[slots[i]].ownership != 2 ||
        bytes[i] != activeBytes(p, slots[i]) ||
        total > p.pinnedReadbackBytes ||
        bytes[i] > p.pinnedReadbackBytes - total) return 1;
    uintptr_t begin = reinterpret_cast<uintptr_t>(outputs[i]);
    if (bytes[i] > UINTPTR_MAX - begin) return 1;
    for (uint32_t j = 0; j < i; ++j) {
      uintptr_t other = reinterpret_cast<uintptr_t>(outputs[j]);
      if (slots[i] == slots[j] ||
          (begin < other + bytes[j] && other < begin + bytes[i])) return 1;
    }
    offsets[i] = total;
    total += bytes[i];
  }
  if (!p.pinned) {
    // Reserve the declared maximum once, avoiding geometric capacity growth
    // beyond the output bytes already charged to the idle-pool budget.
    p.readback.reserve(p.pinnedReadbackBytes);
    p.readback.resize(total);
  }
  void *staging = p.pinned ? p.pinnedReadback : p.readback.data();
  for (uint32_t i = 0; i < count; ++i)
    if (hipMemcpyAsync(static_cast<unsigned char *>(staging) + offsets[i],
                       p.buffers[slots[i]], bytes[i],
                       hipMemcpyDeviceToHost, p.stream) != hipSuccess) {
      p.poisoned = true; return 5;
    }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  for (uint32_t i = 0; i < count; ++i)
    std::memcpy(outputs[i], static_cast<unsigned char *>(staging) + offsets[i], bytes[i]);
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_close(uint64_t handle) try {
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  // Completion and identity precede reuse. A closed handle is always removed;
  // a later acquire gets a fresh handle and retains its generation counter.
  const uint64_t idleLimit = p.idleLimit;
  if (!p.poisoned && cacheEnabled() && p.events.empty() && p.graphs.empty() &&
      p.graphExecutables.empty() &&
      p.allocationBytes <= idleLimit) {
    if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
    // Close appends and acquire removes, so the first idle owner is least
    // recently used. Only completed owners in this device/context may be
    // evicted; failed cleanup remains quarantined and budgeted.
    while (pool.idle.size() >= 4 ||
           !tessera::rocm::programFrameFitsIdleBudget(pool.idleBytes, p.allocationBytes, idleLimit)) {
      auto victim = pool.idle.begin();
      for (; victim != pool.idle.end(); ++victim)
        if (!(*victim)->poisoned && identity(**victim)) break;
      if (victim == pool.idle.end()) break;
      auto &retired = **victim;
      retired.poisoned = true;
      int rc = clean(retired);
      if (rc) continue;
      pool.idleBytes -= retired.allocationBytes;
      pool.idle.erase(victim);
    }
    if (pool.idle.size() >= 4 ||
        !tessera::rocm::programFrameFitsIdleBudget(pool.idleBytes, p.allocationBytes, idleLimit)) {
      p.poisoned = true;
      int rc = clean(p);
      if (!rc) pool.programs.erase(it);
      return rc;
    }
    p.ready = false; p.output = false;
    // Owned host staging is included in allocationBytes; retain its capacity.
    // Private values are not ready until the next complete fresh-root upload.
    p.initialized.assign(p.contract.size(), 0);
    p.readback.clear();
    pool.idleBytes += p.allocationBytes;
    pool.idle.push_back(std::move(it->second));
    pool.programs.erase(it);
    return 0;
  }
  p.poisoned = true;
  int rc = clean(p);
  if (!rc) pool.programs.erase(it);
  return rc;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_cache_stats(
    uint64_t *hits, uint64_t *misses, uint64_t *entries, uint64_t *bytes) try {
  if (!hits || !misses || !entries || !bytes) return 1;
  if (getpid() != process) return 2;
  auto &pool=state(); std::lock_guard<std::mutex> guard(pool.mutex);
  *hits=pool.hits; *misses=pool.misses;
  *entries=pool.idle.size(); *bytes=pool.idleBytes;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_cache_clear() try {
  if (getpid() != process) return 2;
  auto &pool=state(); std::lock_guard<std::mutex> guard(pool.mutex);
  // Foreign-context owners cannot be destroyed from the current context.
  for (auto &p:pool.idle) if (!identity(*p)) return 2;
  while (!pool.idle.empty()) {
    auto &p=*pool.idle.back();
    p.poisoned=true;
    int rc=clean(p); if (rc) return rc;
    pool.idleBytes-=p.allocationBytes;
    pool.idle.pop_back();
  }
  return 0;
} catch (...) { return 12; }
