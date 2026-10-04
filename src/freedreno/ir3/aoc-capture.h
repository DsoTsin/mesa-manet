#ifndef IR3_AOC_CAPTURE_H
#define IR3_AOC_CAPTURE_H

#include <windows.h>
#include <stdexcept>
#include <string>
#include <utility>
#include "aoc-isa.h"

namespace aoc {

inline void require(bool ok, const char *operation)
{
   if (!ok)
      throw std::runtime_error(std::string(operation) + " (Win32 " +
                               std::to_string(GetLastError()) + ")");
}

class Memory {
   HANDLE process_;
public:
   explicit Memory(HANDLE process) : process_(process) {}
   void read(std::uint64_t address, void *output, std::size_t size) const
   {
      SIZE_T copied = 0;
      require(ReadProcessMemory(process_, reinterpret_cast<const void*>(address), output, size, &copied)
              && copied == size, "Read compiler memory");
   }
   template<class T> T read(std::uint64_t address) const
   {
      T value{};
      read(address, &value, sizeof(value));
      return value;
   }
   void enable_isa(const CONTEXT& registers) const
   {
      auto api = read<std::uint32_t>(registers.Rcx + offsetof(Application, api));
      if (api != 1 && api != 2)
         throw std::runtime_error("Only Vulkan SPIR-V and Direct3D DXBC/DXIL inputs are supported");
      std::uint8_t enabled = 1;
      SIZE_T written = 0;
      require(WriteProcessMemory(process_, reinterpret_cast<void*>(registers.Rcx +
                                 offsetof(Application, dump_isa)), &enabled, 1, &written) && written == 1,
              "Enable compiler ISA output");
   }
   void capture(const CONTEXT& registers, std::vector<Capture>& captures) const
   {
      unsigned flags = read<std::uint32_t>(registers.Rsp + 0x38);
      if (!(flags & isa_dump_flag)) return;
      auto span = read<InstructionSpan>(registers.R8);
      if (!span.instruction_count) return;
      if (!span.data || span.instruction_count > 1024 * 1024 || registers.Rcx > 8)
         throw std::runtime_error("Invalid ISA span");
      auto reader = read<std::uint64_t>(registers.Rdx + offsetof(MetadataContextView, reader));
      auto view = read<std::uint64_t>(reader + offsetof(MetadataReaderView, object_span));
      auto object = read<ObjectSpan>(view);
      if (object.byte_count < sizeof(ObjectHeaderView) || object.byte_count > 256 * 1024 * 1024)
         throw std::runtime_error("Invalid shader object span");
      auto header = read<ObjectHeaderView>(object.data);
      std::uint64_t table_size = std::uint64_t(header.section_count) * sizeof(SectionDescriptor);
      if (header.signature != object_magic || header.section_offset > object.byte_count ||
          table_size > object.byte_count - header.section_offset)
         throw std::runtime_error("Invalid shader object section table");
      bool found_code = false, found_flags = false, binning = false;
      for (unsigned i = 0; i < header.section_count; ++i) {
         auto section = read<SectionDescriptor>(object.data + header.section_offset +
                                                i * sizeof(SectionDescriptor));
         if (section.offset > object.byte_count || section.byte_size > object.byte_count - section.offset)
            throw std::runtime_error("Shader object section exceeds its span");
         if (section.type == 10) {
            if (section.element_size != 8 || section.element_count != span.instruction_count ||
                section.byte_size != span.instruction_count * 8 || object.data + section.offset != span.data)
               throw std::runtime_error("ISA span does not match shader object section 10");
            found_code = true;
         }
         if (section.type == 1) {
            if (section.byte_size < 0x24)
               throw std::runtime_error("Shader flags section is truncated");
            binning = (read<std::uint32_t>(object.data + section.offset + 0x20) & binning_flag) != 0;
            found_flags = true;
         }
      }
      if (!found_code || !found_flags)
         throw std::runtime_error("Missing code or shader flags section");
      Capture result{static_cast<unsigned>(registers.Rcx), binning, {}};
      result.words.resize(static_cast<std::size_t>(span.instruction_count));
      read(span.data, result.words.data(), result.words.size() * 8);
      captures.push_back(std::move(result));
   }
};

}
#endif
