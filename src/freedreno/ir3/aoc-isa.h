#ifndef IR3_AOC_ISA_H
#define IR3_AOC_ISA_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aoc {

constexpr std::uint64_t main_rva = 0x1577b0;
constexpr std::uint64_t vulkan_entry_rva = 0x171800;
constexpr std::uint64_t direct3d_entry_rva = 0x170b80;
constexpr std::uint64_t dump_entry_rva = 0x2babf0;
constexpr std::uint32_t object_magic = 0xe74f4751;
constexpr std::uint32_t isa_dump_flag = 8;
constexpr std::uint32_t binning_flag = 0x40;
constexpr std::uint64_t security_cookie_rva = 0x31ad2d0;
constexpr std::uint64_t crt_initialize_rva = 0x31ac79c;
constexpr std::uint64_t onexit_initialize_rva = 0x31ac7e8;
constexpr std::uint64_t onexit_table_rva = 0x4ad06b8;
constexpr std::uint64_t c_initializers_begin_rva = 0x3541da8;
constexpr std::uint64_t c_initializers_end_rva = 0x3541dc8;
constexpr std::uint64_t cpp_initializers_begin_rva = 0x3539d90;
constexpr std::uint64_t cpp_initializers_end_rva = 0x3541d88;
constexpr std::uint64_t tls_index_rva = 0x4ad0698;
constexpr std::uint64_t tls_callbacks_rva = 0x3541dd8;
constexpr std::uint64_t tls_template_rva = 0x4186970;
constexpr std::size_t tls_template_size = 0x188;

struct Capture {
   unsigned stage;
   bool binning;
   std::vector<std::uint64_t> words;
};

struct Application {
   std::uint8_t opaque0[0x300];
   std::uint32_t api;
   std::uint8_t opaque1[0x290];
   std::uint8_t dump_stats;
   std::uint8_t dump_source;
   std::uint8_t dump_isa;
   std::uint8_t dump_detail;
   std::uint8_t opaque2[0x30];
   std::uint8_t json_output;
   std::uint8_t opaque3[0x387];
};

struct InstructionSpan {
   std::uint64_t data;
   std::uint64_t instruction_count;
};

struct ObjectSpan {
   std::uint64_t data;
   std::uint64_t byte_count;
};

struct MetadataContextView {
   std::uint8_t opaque[0x688];
   std::uint64_t reader;
};

struct MetadataReaderView {
   std::uint8_t opaque[0x10];
   std::uint64_t object_span;
};

struct ObjectHeaderView {
   std::uint32_t opaque0;
   std::uint32_t signature;
   std::uint32_t opaque1[3];
   std::uint32_t section_offset;
   std::uint32_t section_count;
};

struct SectionDescriptor {
   std::uint32_t type;
   std::uint32_t offset;
   std::uint32_t byte_size;
   std::uint32_t element_count;
   std::uint32_t element_size;
};

struct ThreadStateView {
   std::uint8_t opaque0[84];
   std::uint8_t initialized;
   std::uint8_t opaque1[11];
   std::uint64_t destructor_list;
   std::uint8_t opaque2[0x120];
};

static_assert(sizeof(Application) == 0x950);
static_assert(offsetof(Application, api) == 0x300);
static_assert(offsetof(Application, dump_isa) == 0x596);
static_assert(offsetof(Application, json_output) == 0x5c8);
static_assert(sizeof(InstructionSpan) == 16);
static_assert(sizeof(ObjectSpan) == 16);
static_assert(offsetof(MetadataContextView, reader) == 0x688);
static_assert(offsetof(MetadataReaderView, object_span) == 0x10);
static_assert(offsetof(ObjectHeaderView, section_offset) == 0x14);
static_assert(offsetof(ObjectHeaderView, section_count) == 0x18);
static_assert(sizeof(SectionDescriptor) == 20);
static_assert(sizeof(ThreadStateView) == tls_template_size);
static_assert(offsetof(ThreadStateView, initialized) == 84);
static_assert(offsetof(ThreadStateView, destructor_list) == 96);

}
#endif
