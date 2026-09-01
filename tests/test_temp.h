#pragma once

#include <cstdlib>
#include <string>

// Writable scratch path for unit tests. `/tmp` is not created on Windows
// GitHub runners (workspace is typically on D:), so fopen("/tmp/...") fails.
inline std::string test_temp_path(const char* name) {
#ifdef _WIN32
    const char* dir = std::getenv("TEMP");
    if (!dir || !dir[0])
        dir = std::getenv("TMP");
    if (!dir || !dir[0])
        dir = ".";
    std::string path = dir;
    const char last = path.back();
    if (last != '\\' && last != '/')
        path += '\\';
    path += name;
    return path;
#else
    return std::string("/tmp/") + name;
#endif
}
