#pragma once

// Narrow POSIX compatibility used by the package loader and optional SSD
// expert cache. Keep this header private to the implementation: the public
// mollm API remains platform-neutral.

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <io.h>
#include <limits>
#include <mutex>
#include <string>
#include <sys/stat.h>

#ifndef _SSIZE_T_DEFINED
using ssize_t = SSIZE_T;
#define _SSIZE_T_DEFINED
#endif

inline int mollm_open_readonly(const char* path) {
    return _open(path, _O_RDONLY | _O_BINARY);
}

inline int mollm_close(int fd) {
    return _close(fd);
}

inline bool mollm_file_size(int fd, uint64_t& size) {
    const __int64 value = _filelengthi64(fd);
    if (value < 0)
        return false;
    size = static_cast<uint64_t>(value);
    return true;
}

inline ssize_t mollm_write(int fd, const void* data, size_t bytes) {
    const unsigned int chunk = static_cast<unsigned int>(
        (std::min<size_t>)(bytes,
                           (std::numeric_limits<unsigned int>::max)()));
    return static_cast<ssize_t>(_write(fd, data, chunk));
}

inline ssize_t mollm_pread(int fd, void* data, size_t bytes, int64_t offset) {
    // The CRT has no positional read. Serialize seek/read pairs so optional
    // SSD worker threads cannot race on the shared descriptor's file pointer.
    static std::mutex read_mutex;
    std::lock_guard<std::mutex> lock(read_mutex);
    if (_lseeki64(fd, offset, SEEK_SET) < 0)
        return -1;
    const unsigned int chunk = static_cast<unsigned int>(
        (std::min<size_t>)(bytes,
                           (std::numeric_limits<unsigned int>::max)()));
    return static_cast<ssize_t>(_read(fd, data, chunk));
}

struct iovec {
    void* iov_base = nullptr;
    size_t iov_len = 0;
};

inline ssize_t mollm_preadv(int fd, const iovec* vectors, int count,
                            int64_t offset) {
    ssize_t total = 0;
    for (int index = 0; index < count; ++index) {
        const ssize_t bytes =
            mollm_pread(fd, vectors[index].iov_base, vectors[index].iov_len,
                        offset);
        if (bytes <= 0)
            return total == 0 ? bytes : total;
        total += bytes;
        offset += bytes;
        if (static_cast<size_t>(bytes) != vectors[index].iov_len)
            break;
    }
    return total;
}

inline int mollm_mkstemp(char* path_template) {
    const size_t length = std::strlen(path_template) + 1;
    if (_mktemp_s(path_template, length) != 0)
        return -1;
    return _open(path_template,
                 _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                 _S_IREAD | _S_IWRITE);
}

inline std::string mollm_default_temp_directory() {
    char buffer[MAX_PATH + 1] = {};
    const DWORD length = GetTempPathA(MAX_PATH, buffer);
    if (length == 0 || length > MAX_PATH)
        return ".";
    return std::string(buffer, length);
}

#define PROT_READ 0x1
#define MAP_PRIVATE 0x2
#define MAP_FAILED reinterpret_cast<void*>(static_cast<intptr_t>(-1))

inline void* mollm_mmap_readonly(int fd, size_t length, int64_t offset = 0) {
    const intptr_t raw_handle = _get_osfhandle(fd);
    if (raw_handle == -1)
        return MAP_FAILED;
    HANDLE mapping = CreateFileMappingA(reinterpret_cast<HANDLE>(raw_handle),
                                        nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping)
        return MAP_FAILED;
    const DWORD offset_high = static_cast<DWORD>(
        (static_cast<uint64_t>(offset) >> 32) & 0xffffffffu);
    const DWORD offset_low =
        static_cast<DWORD>(static_cast<uint64_t>(offset) & 0xffffffffu);
    void* address = MapViewOfFile(mapping, FILE_MAP_READ, offset_high,
                                  offset_low, length);
    CloseHandle(mapping);
    return address ? address : MAP_FAILED;
}

inline int mollm_munmap(void* address, size_t) {
    return address && UnmapViewOfFile(address) ? 0 : -1;
}

inline int mollm_mlock(const void* address, size_t bytes) {
    return address && VirtualLock(const_cast<void*>(address), bytes) ? 0 : -1;
}

inline int mollm_munlock(const void* address, size_t bytes) {
    return address && VirtualUnlock(const_cast<void*>(address), bytes) ? 0 : -1;
}

inline long mollm_page_size() {
    SYSTEM_INFO info {};
    GetSystemInfo(&info);
    return static_cast<long>(info.dwPageSize);
}

#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

inline int mollm_open_readonly(const char* path) {
    return ::open(path, O_RDONLY);
}

inline int mollm_close(int fd) {
    return ::close(fd);
}

inline bool mollm_file_size(int fd, uint64_t& size) {
    struct stat file_stat {};
    if (::fstat(fd, &file_stat) != 0 || file_stat.st_size < 0)
        return false;
    size = static_cast<uint64_t>(file_stat.st_size);
    return true;
}

inline ssize_t mollm_write(int fd, const void* data, size_t bytes) {
    return ::write(fd, data, bytes);
}

inline ssize_t mollm_pread(int fd, void* data, size_t bytes, int64_t offset) {
    return ::pread(fd, data, bytes, static_cast<off_t>(offset));
}

inline ssize_t mollm_preadv(int fd, const iovec* vectors, int count,
                            int64_t offset) {
    return ::preadv(fd, vectors, count, static_cast<off_t>(offset));
}

inline int mollm_mkstemp(char* path_template) {
    return ::mkstemp(path_template);
}

inline void* mollm_mmap_readonly(int fd, size_t length, int64_t offset = 0) {
    return ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd,
                  static_cast<off_t>(offset));
}

inline int mollm_munmap(void* address, size_t bytes) {
    return ::munmap(address, bytes);
}

inline int mollm_mlock(const void* address, size_t bytes) {
    return ::mlock(address, bytes);
}

inline int mollm_munlock(const void* address, size_t bytes) {
    return ::munlock(address, bytes);
}

inline long mollm_page_size() {
    return ::sysconf(_SC_PAGESIZE);
}

#endif
