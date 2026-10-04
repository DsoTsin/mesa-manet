#ifndef IR3_AOC_INPROCESS_H
#define IR3_AOC_INPROCESS_H

#include <windows.h>
#include <string>
#include "aoc-isa.h"

namespace aoc {

struct Compilation {
   DWORD status;
   std::vector<Capture> captures;
};

Compilation compile_in_process(const std::wstring& executable,
                               const std::vector<std::wstring>& arguments, HANDLE log);

}
#endif
