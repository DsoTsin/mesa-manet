#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aoc-isa.h"
#include "ir3-isa.h"

namespace {

class Handle {
   HANDLE value_ = nullptr;
public:
   explicit Handle(HANDLE value = nullptr) : value_(value) {}
   ~Handle() { if (value_ && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
   Handle(const Handle&) = delete;
   Handle& operator=(const Handle&) = delete;
   HANDLE get() const { return value_; }
};

void require(bool ok, const char *operation)
{
   if (!ok)
      throw std::runtime_error(std::string(operation) + " (Win32 " +
                               std::to_string(GetLastError()) + ")");
}

std::wstring quote(const std::wstring& arg)
{
   std::wstring result = L"\"";
   unsigned slashes = 0;
   for (wchar_t ch : arg) {
      if (ch == L'\\') {
         ++slashes;
      } else {
         result.append(slashes * (ch == L'\"' ? 2 : 1), L'\\');
         if (ch == L'\"') result += L'\\';
         result += ch;
         slashes = 0;
      }
   }
   result.append(slashes * 2, L'\\');
   return result + L'\"';
}

std::array<unsigned char, 32> sha256(HANDLE file)
{
   BCRYPT_ALG_HANDLE algorithm = nullptr;
   BCRYPT_HASH_HANDLE hash = nullptr;
   if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
      throw std::runtime_error("Cannot open SHA-256 provider");
   struct Cleanup {
      BCRYPT_ALG_HANDLE& algorithm;
      BCRYPT_HASH_HANDLE& hash;
      ~Cleanup() { if (hash) BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0); }
   } cleanup{algorithm, hash};
   if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0)
      throw std::runtime_error("Cannot create SHA-256 hash");
   std::array<unsigned char, 65536> bytes{};
   DWORD size;
   do {
      require(ReadFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &size, nullptr), "Read AOC");
      if (BCryptHashData(hash, bytes.data(), size, 0) < 0)
         throw std::runtime_error("Cannot hash AOC");
   } while (size);
   std::array<unsigned char, 32> result{};
   if (BCryptFinishHash(hash, result.data(), static_cast<ULONG>(result.size()), 0) < 0)
      throw std::runtime_error("Cannot finish SHA-256 hash");
   return result;
}

struct Capture {
   unsigned stage;
   bool binning;
   std::vector<std::uint64_t> words;
};

class TemporaryInputs {
   std::vector<std::filesystem::path> files_;
   unsigned serial_ = 0;

   std::wstring copy(const std::filesystem::path& source, const std::wstring& suffix)
   {
      auto name = L"aoc-isa-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                  std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(serial_++) + suffix;
      auto destination = std::filesystem::current_path() / name;
      Handle output(CreateFileW(destination.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                FILE_ATTRIBUTE_TEMPORARY, nullptr));
      require(output.get() != INVALID_HANDLE_VALUE, "Create compiler input copy");
      files_.push_back(destination);
      Handle input(CreateFileW(source.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
      require(input.get() != INVALID_HANDLE_VALUE, "Open compiler input copy");
      char bytes[65536];
      DWORD size = 0;
      do {
         require(ReadFile(input.get(), bytes, sizeof(bytes), &size, nullptr), "Read compiler input copy");
         DWORD written = 0;
         require(WriteFile(output.get(), bytes, size, &written, nullptr) && written == size,
                 "Write compiler input copy");
      } while (size);
      return name;
   }

public:
   ~TemporaryInputs() { for (const auto& file : files_) DeleteFileW(file.c_str()); }

   void prepare(std::vector<std::wstring>& args)
   {
      for (std::size_t i = 0; i < args.size(); ++i) {
         if (args[i] == L"-link_info" && i + 1 < args.size()) {
            std::filesystem::path source(args[++i]);
            if (!source.parent_path().empty() && std::filesystem::is_regular_file(source))
               args[i] = copy(source, L".json");
            continue;
         }
         std::filesystem::path source(args[i]);
         auto extension = source.extension().wstring();
         if ((extension == L".dxbc" || extension == L".dxil") &&
             args[i].find_first_of(L" \t") != std::wstring::npos &&
             std::filesystem::is_regular_file(source))
            args[i] = copy(source, source.stem().extension().wstring() + extension);
      }
   }
};

class CompilerProcess {
   HANDLE process_ = nullptr;
   std::map<DWORD, HANDLE> threads_;
   std::uint64_t image_base_ = 0;
   bool active_ = false;
   bool initial_breakpoint_ = false;
   std::vector<Capture> captures_;

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

   void arm(HANDLE thread) const
   {
      CONTEXT registers{};
      registers.ContextFlags = CONTEXT_DEBUG_REGISTERS;
      require(GetThreadContext(thread, &registers), "Read debug registers");
      registers.Dr0 = image_base_ + aoc::vulkan_entry_rva;
      registers.Dr1 = image_base_ + aoc::direct3d_entry_rva;
      registers.Dr2 = image_base_ + aoc::dump_entry_rva;
      registers.Dr3 = 0;
      registers.Dr6 = 0;
      registers.Dr7 = 0x15;
      require(SetThreadContext(thread, &registers), "Set hardware breakpoints");
   }

   void add_thread(DWORD id, HANDLE event_handle)
   {
      HANDLE owned = nullptr;
      require(DuplicateHandle(GetCurrentProcess(), event_handle, GetCurrentProcess(), &owned,
                              0, FALSE, DUPLICATE_SAME_ACCESS), "Own compiler thread handle");
      threads_[id] = owned;
      arm(owned);
   }

   void capture(const CONTEXT& registers)
   {
      unsigned flags = read<std::uint32_t>(registers.Rsp + 0x38);
      if (!(flags & aoc::isa_dump_flag)) return;
      auto span = read<aoc::InstructionSpan>(registers.R8);
      if (!span.instruction_count) return;
      if (!span.data || span.instruction_count > 1024 * 1024 || registers.Rcx > 8)
         throw std::runtime_error("Invalid ISA span");
      auto reader = read<std::uint64_t>(registers.Rdx + offsetof(aoc::MetadataContextView, reader));
      auto view = read<std::uint64_t>(reader + offsetof(aoc::MetadataReaderView, object_span));
      auto object = read<aoc::ObjectSpan>(view);
      if (object.byte_count < sizeof(aoc::ObjectHeaderView) || object.byte_count > 256 * 1024 * 1024)
         throw std::runtime_error("Invalid shader object span");
      auto header = read<aoc::ObjectHeaderView>(object.data);
      std::uint64_t table_size = std::uint64_t(header.section_count) * sizeof(aoc::SectionDescriptor);
      if (header.signature != aoc::object_magic || header.section_offset > object.byte_count ||
          table_size > object.byte_count - header.section_offset)
         throw std::runtime_error("Invalid shader object section table");
      bool found_code = false;
      bool found_flags = false;
      bool binning = false;
      for (unsigned i = 0; i < header.section_count; ++i) {
         auto section = read<aoc::SectionDescriptor>(object.data + header.section_offset +
                                                     i * sizeof(aoc::SectionDescriptor));
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
            binning = (read<std::uint32_t>(object.data + section.offset + 0x20) & aoc::binning_flag) != 0;
            found_flags = true;
         }
      }
      if (!found_code || !found_flags)
         throw std::runtime_error("Missing code or shader flags section");
      Capture result{static_cast<unsigned>(registers.Rcx), binning, {}};
      result.words.resize(static_cast<std::size_t>(span.instruction_count));
      read(span.data, result.words.data(), result.words.size() * 8);
      captures_.push_back(std::move(result));
   }

   void breakpoint(DWORD thread_id)
   {
      auto it = threads_.find(thread_id);
      if (it == threads_.end()) throw std::runtime_error("Unknown compiler thread");
      CONTEXT registers{};
      registers.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_DEBUG_REGISTERS;
      require(GetThreadContext(it->second, &registers), "Read compiler registers");
      if (registers.Rip == image_base_ + aoc::vulkan_entry_rva ||
          registers.Rip == image_base_ + aoc::direct3d_entry_rva) {
         auto api = read<std::uint32_t>(registers.Rcx + offsetof(aoc::Application, api));
         if (api != 1 && api != 2)
            throw std::runtime_error("Only Vulkan SPIR-V and Direct3D DXBC/DXIL inputs are supported");
         std::uint8_t enabled = 1;
         SIZE_T written = 0;
         require(WriteProcessMemory(process_, reinterpret_cast<void*>(registers.Rcx +
                                    offsetof(aoc::Application, dump_isa)), &enabled, 1, &written) && written == 1,
                 "Enable compiler ISA output");
      } else if (registers.Rip == image_base_ + aoc::dump_entry_rva) {
         capture(registers);
      } else {
         throw std::runtime_error("Unexpected compiler hardware breakpoint");
      }
      registers.Dr6 = 0;
      registers.EFlags |= 0x10000;
      require(SetThreadContext(it->second, &registers), "Resume compiler instruction");
   }

public:
   ~CompilerProcess()
   {
      if (active_) TerminateProcess(process_, 1);
      for (auto& entry : threads_) CloseHandle(entry.second);
      if (process_) CloseHandle(process_);
   }

   DWORD run(const std::wstring& executable, const std::vector<std::wstring>& arguments, HANDLE log)
   {
      std::wstring command = quote(executable);
      for (const auto& arg : arguments) command += L" " + quote(arg);
      SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
      Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
      require(input.get() != INVALID_HANDLE_VALUE, "Open compiler input");
      STARTUPINFOW startup{};
      startup.cb = sizeof(startup);
      startup.dwFlags = STARTF_USESTDHANDLES;
      startup.hStdInput = input.get();
      startup.hStdOutput = log;
      startup.hStdError = log;
      PROCESS_INFORMATION info{};
      require(CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                             DEBUG_ONLY_THIS_PROCESS | CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info),
              "Start compiler");
      process_ = info.hProcess;
      CloseHandle(info.hThread);
      active_ = true;
      auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
      while (active_) {
         DEBUG_EVENT event{};
         if (!WaitForDebugEvent(&event, 200)) {
            if (GetLastError() != ERROR_SEM_TIMEOUT) require(false, "Wait for compiler");
            if (std::chrono::steady_clock::now() > deadline)
               throw std::runtime_error("Compiler exceeded 120 seconds");
            continue;
         }
         DWORD disposition = DBG_CONTINUE;
         try {
            switch (event.dwDebugEventCode) {
            case CREATE_PROCESS_DEBUG_EVENT:
               image_base_ = reinterpret_cast<std::uint64_t>(event.u.CreateProcessInfo.lpBaseOfImage);
               if (event.u.CreateProcessInfo.hFile) CloseHandle(event.u.CreateProcessInfo.hFile);
               add_thread(event.dwThreadId, event.u.CreateProcessInfo.hThread);
               break;
            case CREATE_THREAD_DEBUG_EVENT:
               add_thread(event.dwThreadId, event.u.CreateThread.hThread);
               break;
            case EXIT_THREAD_DEBUG_EVENT:
               if (threads_.count(event.dwThreadId)) {
                  CloseHandle(threads_[event.dwThreadId]);
                  threads_.erase(event.dwThreadId);
               }
               break;
            case LOAD_DLL_DEBUG_EVENT:
               if (event.u.LoadDll.hFile) CloseHandle(event.u.LoadDll.hFile);
               break;
            case EXCEPTION_DEBUG_EVENT:
               if (event.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_SINGLE_STEP) {
                  breakpoint(event.dwThreadId);
               } else if (event.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_BREAKPOINT &&
                          !initial_breakpoint_) {
                  initial_breakpoint_ = true;
               } else {
                  disposition = DBG_EXCEPTION_NOT_HANDLED;
               }
               break;
            case EXIT_PROCESS_DEBUG_EVENT:
               active_ = false;
               require(ContinueDebugEvent(event.dwProcessId, event.dwThreadId, disposition), "Finish compiler");
               return event.u.ExitProcess.dwExitCode;
            }
         } catch (...) {
            TerminateProcess(process_, 1);
            ContinueDebugEvent(event.dwProcessId, event.dwThreadId, DBG_CONTINUE);
            active_ = false;
            throw;
         }
         require(ContinueDebugEvent(event.dwProcessId, event.dwThreadId, disposition), "Continue compiler");
      }
      return 1;
   }

   const std::vector<Capture>& captures() const { return captures_; }
};

unsigned unmatched = 0;

void print_instruction(void *data, unsigned index, void *instruction)
{
   std::uint32_t words[2];
   std::memcpy(words, instruction, sizeof(words));
   std::fprintf(static_cast<FILE*>(data), "%5u[%08x_%08x] ", index, words[1], words[0]);
}

void print_unknown(FILE *out, const BITSET_WORD *bits, std::size_t)
{
   ++unmatched;
   std::uint32_t words[2];
   std::memcpy(words, bits, sizeof(words));
   std::fprintf(out, ".quad 0x%08x%08x ; unmatched instruction", words[1], words[0]);
}

unsigned gpu_generation(const std::vector<std::wstring>& args)
{
   unsigned gpu = 600;
   for (auto arg : args) {
      for (auto& ch : arg) if (ch >= L'A' && ch <= L'Z') ch += L'a' - L'A';
      if (arg.rfind(L"-arch=", 0) != 0) continue;
      arg = arg.substr(6);
      if (arg == L"a830" || arg == L"a840" || arg == L"d500" || arg == L"d501" ||
          arg == L"d510" || arg == L"d511") gpu = 800;
      else if (arg == L"a730" || arg == L"a740" || arg == L"a741" || arg == L"a750" ||
               arg == L"c510" || arg == L"c511" || arg == L"c512" || arg == L"c520") gpu = 700;
      else if (arg == L"a608" || arg == L"a640" || arg == L"a650" || arg == L"a660" || arg == L"a690") gpu = 600;
      else throw std::runtime_error("Unsupported AOC architecture");
   }
   return gpu;
}

void usage()
{
   std::fprintf(stderr, "Usage: ir3-aoc-isa [--aoc PATH] [--output FILE] [--raw-dir DIR]\n"
                        "                   [--aoc-log FILE] -- AOC_ARGUMENTS\n"
                        "Examples: -- -arch=a830 shader.comp.spv\n"
                        "          -- -arch=a741 -link_info pipeline.json\n");
}

}

int wmain(int argc, wchar_t **argv)
{
   try {
      static_assert(sizeof(void*) == 8);
      std::wstring executable = L"C:\\Program Files\\Qualcomm\\Adreno Offline Compiler\\aoc.exe";
      std::filesystem::path output, raw_dir, aoc_log;
      std::vector<std::wstring> args;
      bool separator = false;
      for (int i = 1; i < argc; ++i) {
         std::wstring arg = argv[i];
         if (separator) { args.push_back(arg); continue; }
         if (arg == L"--") { separator = true; continue; }
         if (arg == L"--help" || arg == L"-h") { usage(); return 0; }
         if (i + 1 >= argc) { usage(); return 1; }
         if (arg == L"--aoc") executable = argv[++i];
         else if (arg == L"--output") output = argv[++i];
         else if (arg == L"--raw-dir") raw_dir = argv[++i];
         else if (arg == L"--aoc-log") aoc_log = argv[++i];
         else { usage(); return 1; }
      }
      if (args.empty()) { usage(); return 1; }
      unsigned gpu = gpu_generation(args);
      executable = std::filesystem::absolute(executable).wstring();
      Handle vendor(CreateFileW(executable.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr));
      require(vendor.get() != INVALID_HANDLE_VALUE, "Open installed compiler");
      const std::array<unsigned char, 32> expected = {
         0x10,0x0e,0xeb,0xad,0xdf,0xb0,0xf1,0x25,0xee,0x7a,0x7e,0x1e,0x62,0x1d,0x56,0xf3,
         0x2e,0xcb,0xa7,0xa1,0x5f,0x80,0xc2,0xe4,0x54,0x61,0xca,0x5a,0x94,0x1f,0x56,0x48
      };
      if (sha256(vendor.get()) != expected)
         throw std::runtime_error("AOC SHA-256 differs from analyzed 7.0.15 binary; private ABI disabled");
      TemporaryInputs temporary_inputs;
      temporary_inputs.prepare(args);
      wchar_t temporary_path[MAX_PATH], log_path[MAX_PATH];
      require(GetTempPathW(MAX_PATH, temporary_path) != 0, "Find temporary directory");
      require(GetTempFileNameW(temporary_path, L"aoc", 0, log_path) != 0, "Create compiler log");
      SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
      Handle log(CreateFileW(log_path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, &inherit,
                             OPEN_EXISTING, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr));
      require(log.get() != INVALID_HANDLE_VALUE, "Open compiler log");
      CompilerProcess compiler;
      DWORD status = compiler.run(executable, args, log.get());
      LARGE_INTEGER origin{};
      require(SetFilePointerEx(log.get(), origin, nullptr, FILE_BEGIN), "Read compiler log");
      std::string diagnostic;
      char chunk[8192];
      DWORD length;
      do {
         require(ReadFile(log.get(), chunk, sizeof(chunk), &length, nullptr), "Read compiler output");
         diagnostic.append(chunk, length);
      } while (length);
      if (!aoc_log.empty()) {
         FILE *file = _wfopen(aoc_log.c_str(), L"wb");
         if (!file) throw std::runtime_error("Cannot open AOC log output");
         auto written = std::fwrite(diagnostic.data(), 1, diagnostic.size(), file);
         bool failed = std::fclose(file) != 0 || written != diagnostic.size();
         if (failed) throw std::runtime_error("Cannot write AOC log output");
      }
      bool success = diagnostic.find("Compilation succeeded") != std::string::npos &&
                     diagnostic.find("Compilation failed") == std::string::npos;
      if (status || !success || compiler.captures().empty()) {
         std::fwrite(diagnostic.data(), 1, diagnostic.size(), stderr);
         if (status) std::fprintf(stderr, "AOC process status: 0x%08lx\n", static_cast<unsigned long>(status));
         throw std::runtime_error("AOC compilation did not produce a successful ISA capture");
      }
      FILE *text = output.empty() ? stdout : _wfopen(output.c_str(), L"wb");
      if (!text) throw std::runtime_error("Cannot open ISA output file");
      struct OutputCleanup { FILE *file; ~OutputCleanup() { if (file != stdout) std::fclose(file); } } cleanup{text};
      if (!raw_dir.empty()) std::filesystem::create_directories(raw_dir);
      static const char *stages[] = {"VS", "TCS", "TES", "GS", "FS", "CS", "BVH", "RAY", "TASK"};
      unsigned index = 0;
      for (const auto& capture : compiler.captures()) {
         std::fprintf(text, "; AOC 7.0.15 GPU %u %s%s, %zu instructions\n", gpu,
                      capture.binning ? "BINNING " : "", stages[capture.stage], capture.words.size());
         isa_decode_options options{};
         options.gpu_id = gpu;
         options.show_errors = true;
         options.branch_labels = true;
         options.cbdata = text;
         options.pre_instr_cb = print_instruction;
         options.no_match_cb = print_unknown;
         ir3_isa_disasm(const_cast<std::uint64_t*>(capture.words.data()),
                        static_cast<int>(capture.words.size() * 8), text, &options);
         std::fprintf(text, "\n");
         if (!raw_dir.empty()) {
            auto path = raw_dir / (std::to_wstring(index) + L"-" +
                                  std::wstring(stages[capture.stage], stages[capture.stage] +
                                               std::strlen(stages[capture.stage])) +
                                  (capture.binning ? L"-binning.bin" : L".bin"));
            FILE *file = _wfopen(path.c_str(), L"wb");
            if (!file) throw std::runtime_error("Cannot open raw ISA output");
            std::size_t written = std::fwrite(capture.words.data(), 8, capture.words.size(), file);
            bool failed = std::fclose(file) != 0 || written != capture.words.size();
            if (failed) throw std::runtime_error("Cannot write raw ISA output");
         }
         ++index;
      }
      if (std::fflush(text) || std::ferror(text)) throw std::runtime_error("Cannot write ISA text");
      std::fprintf(stderr, "Captured %u shader variants; %u unmatched instructions\n", index, unmatched);
      return unmatched ? 2 : 0;
   } catch (const std::exception& error) {
      std::fprintf(stderr, "ir3-aoc-isa: %s\n", error.what());
      return 1;
   }
}
