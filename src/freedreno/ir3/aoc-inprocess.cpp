#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <corecrt_startup.h>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "aoc-capture.h"
#include "aoc-inprocess.h"

namespace aoc {
namespace {

class OutputRedirect {
   int saved_[2] = {-1, -1};
   HANDLE handles_[2]{};
   bool descriptor_handles_[2]{};
   int mode_[2] = {-1, -1};
   int target_ = -1;
public:
   explicit OutputRedirect(HANDLE log)
   {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      HANDLE owned = nullptr;
      require(DuplicateHandle(GetCurrentProcess(), log, GetCurrentProcess(), &owned,
                              0, FALSE, DUPLICATE_SAME_ACCESS), "Own compiler log handle");
      target_ = _open_osfhandle(reinterpret_cast<intptr_t>(owned), _O_WRONLY | _O_BINARY);
      if (target_ < 0) { CloseHandle(owned); throw std::runtime_error("Open compiler log descriptor"); }
      for (int i = 0; i < 2; ++i) {
         saved_[i] = _dup(i + 1);
         handles_[i] = GetStdHandle(i ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
         descriptor_handles_[i] = handles_[i] == reinterpret_cast<HANDLE>(_get_osfhandle(i + 1));
         if (saved_[i] < 0) { restore(); throw std::runtime_error("Save standard output descriptor"); }
         mode_[i] = _setmode(i + 1, _O_BINARY);
      }
      for (int i = 0; i < 2; ++i) {
         if (_dup2(target_, i + 1)) { restore(); throw std::runtime_error("Redirect compiler output"); }
         if (!SetStdHandle(i ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE, owned)) {
            restore();
            require(false, "Redirect standard output handle");
         }
      }
   }
   void restore()
   {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      for (int i = 0; i < 2; ++i) {
         if (saved_[i] < 0) continue;
         _dup2(saved_[i], i + 1);
         if (mode_[i] >= 0) _setmode(i + 1, mode_[i]);
         SetStdHandle(i ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE, descriptor_handles_[i] ?
                      reinterpret_cast<HANDLE>(_get_osfhandle(i + 1)) : handles_[i]);
         _close(saved_[i]);
         saved_[i] = -1;
      }
      if (target_ >= 0) { _close(target_); target_ = -1; }
   }
   ~OutputRedirect() { restore(); }
   OutputRedirect(const OutputRedirect&) = delete;
   OutputRedirect& operator=(const OutputRedirect&) = delete;
};

class Image {
   HMODULE image_ = nullptr;
   HMODULE tls_ = nullptr;
   std::vector<HMODULE> imports_;
   void (*notify_)(DWORD) = nullptr;
   void (*unbind_)() = nullptr;
   int (*execute_onexit_)(_onexit_table_t*) = nullptr;
   bool initialized_ = false;
   bool tls_enabled_ = false;

   template<class T> T function(std::uint64_t rva) const
   {
      return reinterpret_cast<T>(base() + rva);
   }
   template<class T> T symbol(const char *name) const
   {
      auto result = GetProcAddress(tls_, name);
      require(result != nullptr, name);
      return reinterpret_cast<T>(result);
   }
   void imports()
   {
      auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image_);
      auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base() + dos->e_lfanew);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE || nt->Signature != IMAGE_NT_SIGNATURE ||
          nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64 ||
          nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
         throw std::runtime_error("AOC is not an AMD64 PE image");
      const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
      auto descriptors = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base() + directory.VirtualAddress);
      unsigned resolved = 0;
      for (auto item = descriptors; item->Name; ++item) {
         auto module = LoadLibraryExA(reinterpret_cast<const char*>(base() + item->Name), nullptr,
                                     LOAD_LIBRARY_SEARCH_SYSTEM32);
         require(module != nullptr, "Load compiler import library");
         imports_.push_back(module);
         auto names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base() + item->OriginalFirstThunk);
         auto slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base() + item->FirstThunk);
         for (; names->u1.AddressOfData; ++names, ++slots) {
            LPCSTR name = IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal) ?
               MAKEINTRESOURCEA(IMAGE_ORDINAL64(names->u1.Ordinal)) :
               reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base() + names->u1.AddressOfData)->Name;
            auto proc = GetProcAddress(module, name);
            require(proc != nullptr, "Resolve compiler import");
            DWORD protection, previous;
            require(VirtualProtect(slots, sizeof(*slots), PAGE_READWRITE, &protection), "Write mapped compiler IAT");
            slots->u1.Function = reinterpret_cast<ULONG_PTR>(proc);
            require(VirtualProtect(slots, sizeof(*slots), protection, &previous), "Protect mapped compiler IAT");
            ++resolved;
         }
      }
      if (resolved != 409) throw std::runtime_error("Unexpected compiler import count");
      auto crt = GetModuleHandleW(L"ucrtbase.dll");
      require(crt != nullptr, "Find compiler CRT");
      execute_onexit_ = reinterpret_cast<int(*)(_onexit_table_t*)>(GetProcAddress(crt, "_execute_onexit_table"));
      require(execute_onexit_ != nullptr, "Find compiler onexit executor");
   }

public:
   Image() = default;
   Image(const Image&) = delete;
   Image& operator=(const Image&) = delete;
   std::uint64_t base() const { return reinterpret_cast<std::uint64_t>(image_); }
   void initialize(const std::wstring& executable)
   {
      image_ = LoadLibraryW(executable.c_str());
      require(image_ != nullptr, "Load installed compiler image");
      imports();
      wchar_t path[32768];
      auto length = GetModuleFileNameW(nullptr, path, std::size(path));
      require(length && length < std::size(path), "Find TLS carrier directory");
      auto carrier = std::filesystem::path(std::wstring(path, length)).parent_path() / L"ir3-aoc-tls.dll";
      tls_ = LoadLibraryExW(carrier.c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
      require(tls_ != nullptr, "Load compiler TLS carrier");
      auto index = symbol<DWORD(*)()>("aoc_tls_get_index")();
      *reinterpret_cast<DWORD*>(base() + tls_index_rva) = index;
      auto bind = symbol<BOOL(*)(HMODULE, PIMAGE_TLS_CALLBACK*, const void*, SIZE_T)>("aoc_tls_bind");
      notify_ = symbol<void(*)(DWORD)>("aoc_tls_notify");
      unbind_ = symbol<void(*)()>("aoc_tls_unbind");
      require(bind(image_, reinterpret_cast<PIMAGE_TLS_CALLBACK*>(base() + tls_callbacks_rva),
                   reinterpret_cast<const void*>(base() + tls_template_rva), tls_template_size), "Bind compiler TLS");
      function<void(*)()>(security_cookie_rva)();
      if (!function<bool(*)(int)>(crt_initialize_rva)(1)) throw std::runtime_error("Initialize compiler CRT");
      if (!function<bool(*)(unsigned)>(onexit_initialize_rva)(0)) throw std::runtime_error("Initialize private compiler onexit tables");
      initialized_ = true;
      for (auto entry = base() + c_initializers_begin_rva; entry < base() + c_initializers_end_rva; entry += 8) {
         auto callback = *reinterpret_cast<int(**)()>(entry);
         if (callback && callback()) throw std::runtime_error("Compiler C initializer failed");
      }
      for (auto entry = base() + cpp_initializers_begin_rva; entry < base() + cpp_initializers_end_rva; entry += 8) {
         auto callback = *reinterpret_cast<void(**)()>(entry);
         if (callback) callback();
      }
      auto table = reinterpret_cast<const _onexit_table_t*>(base() + onexit_table_rva);
      if (table->_first == reinterpret_cast<_PVFV*>(-1) || table->_first == table->_last)
         throw std::runtime_error("Compiler destructors were not registered in its private table");
      notify_(DLL_THREAD_ATTACH);
      tls_enabled_ = true;
   }
   void finish()
   {
      static_assert(sizeof(_onexit_table_t) == 24);
      if (tls_enabled_) {
         notify_(DLL_THREAD_DETACH);
         tls_enabled_ = false;
         auto current = symbol<void*(*)()>("aoc_tls_current")();
         if (*reinterpret_cast<const std::uint64_t*>(static_cast<char*>(current) + offsetof(ThreadStateView, destructor_list)))
            throw std::runtime_error("Compiler thread destructor list remained populated");
      }
      if (initialized_) {
         auto table = reinterpret_cast<_onexit_table_t*>(base() + onexit_table_rva);
         int status = execute_onexit_(table);
         initialized_ = false;
         if (status || table->_first != table->_last || table->_first != table->_end)
            throw std::runtime_error("Compiler private destructor table was not released");
      }
   }
   int main(int argc, char **argv) const
   {
      return function<int(*)(int, char**, char**)>(main_rva)(argc, argv, nullptr);
   }
   ~Image()
   {
      if (tls_enabled_) notify_(DLL_THREAD_DETACH);
      if (initialized_) execute_onexit_(reinterpret_cast<_onexit_table_t*>(base() + onexit_table_rva));
      if (unbind_) unbind_();
      if (tls_) FreeLibrary(tls_);
      if (image_) FreeLibrary(image_);
      for (auto it = imports_.rbegin(); it != imports_.rend(); ++it) FreeLibrary(*it);
   }
};

class Worker {
   const Image& image_;
   std::vector<std::string> strings_;
   std::vector<char*> argv_;
   Compilation result_{};
   std::exception_ptr error_;
   DWORD id_ = 0;
   static Worker *active_;

   static LONG CALLBACK exception(EXCEPTION_POINTERS *pointers)
   {
      auto self = active_;
      if (!self || GetCurrentThreadId() != self->id_ ||
          pointers->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
         return EXCEPTION_CONTINUE_SEARCH;
      auto& registers = *pointers->ContextRecord;
      auto base = self->image_.base();
      bool branch = registers.Rip == base + vulkan_entry_rva || registers.Rip == base + direct3d_entry_rva;
      if (!branch && registers.Rip != base + dump_entry_rva) return EXCEPTION_CONTINUE_SEARCH;
      try {
         Memory memory(GetCurrentProcess());
         if (branch) memory.enable_isa(registers);
         else memory.capture(registers, self->result_.captures);
      } catch (...) {
         self->error_ = std::current_exception();
         registers.Dr7 = 0;
      }
      registers.Dr6 = 0;
      registers.EFlags |= 0x10000;
      return EXCEPTION_CONTINUE_EXECUTION;
   }
   static DWORD WINAPI entry(void *argument)
   {
      auto self = static_cast<Worker*>(argument);
      try {
         return static_cast<DWORD>(self->image_.main(static_cast<int>(self->argv_.size() - 1),
                                                     self->argv_.data()));
      } catch (...) {
         self->error_ = std::current_exception();
         return 1;
      }
   }
   static std::string narrow(const std::wstring& text)
   {
      if (text.empty()) return {};
      UINT page = GetACP();
      DWORD flags = page == CP_UTF8 ? WC_ERR_INVALID_CHARS : WC_NO_BEST_FIT_CHARS;
      int size = WideCharToMultiByte(page, flags, text.data(),
                                     static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
      require(size != 0, "Convert AOC argument");
      std::string result(size, '\0');
      BOOL substituted = FALSE;
      require(WideCharToMultiByte(page, flags, text.data(), static_cast<int>(text.size()),
                                  result.data(), size, nullptr, page == CP_UTF8 ? nullptr : &substituted) != 0,
              "Convert AOC argument");
      if (substituted) throw std::runtime_error("AOC argument cannot be represented in the Windows ANSI code page");
      return result;
   }
public:
   explicit Worker(const Image& image, const std::vector<std::wstring>& arguments) : image_(image)
   {
      strings_.push_back("aoc.exe");
      for (const auto& argument : arguments) strings_.push_back(narrow(argument));
      for (auto& value : strings_) argv_.push_back(value.data());
      argv_.push_back(nullptr);
   }
   Compilation run()
   {
      if (active_) throw std::runtime_error("Concurrent in-process compiler use is unsupported");
      active_ = this;
      auto handler = AddVectoredExceptionHandler(1, exception);
      if (!handler) { active_ = nullptr; require(false, "Register compiler exception handler"); }
      struct HandlerCleanup {
         void *handler;
         ~HandlerCleanup() { RemoveVectoredExceptionHandler(handler); active_ = nullptr; }
      } cleanup{handler};
      auto thread = CreateThread(nullptr, 0, entry, this, CREATE_SUSPENDED, &id_);
      require(thread != nullptr, "Create compiler worker thread");
      struct ThreadCleanup { HANDLE handle; ~ThreadCleanup() { CloseHandle(handle); } } thread_cleanup{thread};
      CONTEXT registers{};
      registers.ContextFlags = CONTEXT_DEBUG_REGISTERS;
      registers.Dr0 = image_.base() + vulkan_entry_rva;
      registers.Dr1 = image_.base() + direct3d_entry_rva;
      registers.Dr2 = image_.base() + dump_entry_rva;
      registers.Dr7 = 0x15;
      if (!SetThreadContext(thread, &registers)) {
         auto failure = GetLastError();
         ResumeThread(thread);
         WaitForSingleObject(thread, INFINITE);
         SetLastError(failure);
         require(false, "Set in-process compiler hardware breakpoints");
      }
      require(ResumeThread(thread) != static_cast<DWORD>(-1), "Start compiler worker");
      require(WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0, "Join compiler worker");
      require(GetExitCodeThread(thread, &result_.status), "Read compiler status");
      if (error_) std::rethrow_exception(error_);
      return std::move(result_);
   }
};

Worker *Worker::active_ = nullptr;

}

Compilation compile_in_process(const std::wstring& executable,
                               const std::vector<std::wstring>& arguments, HANDLE log)
{
   OutputRedirect redirect(log);
   Image image;
   image.initialize(executable);
   Worker worker(image, arguments);
   auto result = worker.run();
   image.finish();
   return result;
}

}
