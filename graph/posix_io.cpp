#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#endif

#include "graph/posix_io.h"

#include <cerrno>
#include <cstring>
#include <limits>

#if defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <cstdlib>
#include <sys/resource.h>
#endif

namespace mollm::io {
namespace {

#if defined(_WIN32)
HANDLE handle_from_fd(int fd) {
    return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}

void set_errno_from_win() {
    switch (GetLastError()) {
    case ERROR_ACCESS_DENIED:
        errno = EACCES;
        break;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
        errno = ENOENT;
        break;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
        errno = ENOMEM;
        break;
    case ERROR_HANDLE_EOF:
        errno = 0;
        break;
    default:
        errno = EIO;
        break;
    }
}
#endif

}  // namespace

int open_read(const char* path) {
#if defined(_WIN32)
    const int fd = _open(path, _O_RDONLY | _O_BINARY);
    return fd;
#else
    return ::open(path, O_RDONLY);
#endif
}

int close(int fd) {
#if defined(_WIN32)
    return _close(fd);
#else
    return ::close(fd);
#endif
}

int write(int fd, const void* data, size_t length) {
#if defined(_WIN32)
    if (length > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
        length = static_cast<size_t>(std::numeric_limits<unsigned int>::max());
    return _write(fd, data, static_cast<unsigned int>(length));
#else
    return static_cast<int>(::write(fd, data, length));
#endif
}

int file_size(int fd, uint64_t* size) {
    if (!size)
        return -1;
#if defined(_WIN32)
    struct _stat64 info;
    if (_fstat64(fd, &info) != 0)
        return -1;
    if (info.st_size < 0)
        return -1;
    *size = static_cast<uint64_t>(info.st_size);
    return 0;
#else
    struct stat info;
    if (::fstat(fd, &info) != 0 || info.st_size < 0)
        return -1;
    *size = static_cast<uint64_t>(info.st_size);
    return 0;
#endif
}

ssize_t pread(int fd, void* buffer, size_t length, uint64_t offset) {
#if defined(_WIN32)
    HANDLE handle = handle_from_fd(fd);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    if (length > 0x40000000u)
        length = 0x40000000u;
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset & 0xffffffffu);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD nread = 0;
    if (!ReadFile(handle, buffer, static_cast<DWORD>(length), &nread,
                  &overlapped)) {
        if (GetLastError() == ERROR_HANDLE_EOF)
            return 0;
        set_errno_from_win();
        return -1;
    }
    return static_cast<ssize_t>(nread);
#else
    return ::pread(fd, buffer, length, static_cast<off_t>(offset));
#endif
}

ssize_t preadv(int fd, const iovec* vectors, int count, uint64_t offset) {
#if defined(_WIN32)
    ssize_t total = 0;
    for (int i = 0; i < count; ++i) {
        size_t remaining = vectors[i].iov_len;
        auto* cursor = static_cast<uint8_t*>(vectors[i].iov_base);
        while (remaining > 0) {
            const ssize_t n = pread(fd, cursor, remaining, offset);
            if (n < 0)
                return total == 0 ? n : total;
            if (n == 0)
                return total;
            total += n;
            offset += static_cast<uint64_t>(n);
            cursor += n;
            remaining -= static_cast<size_t>(n);
        }
    }
    return total;
#else
    return ::preadv(fd, vectors, count, static_cast<off_t>(offset));
#endif
}

void* mmap(void* address, size_t length, int prot, int flags, int fd,
           uint64_t offset) {
    (void)prot;
    (void)flags;
#if defined(_WIN32)
    if (address != nullptr) {
        errno = EINVAL;
        return reinterpret_cast<void*>(-1);
    }
    HANDLE file = handle_from_fd(fd);
    if (file == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return reinterpret_cast<void*>(-1);
    }
    HANDLE mapping =
        CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        set_errno_from_win();
        return reinterpret_cast<void*>(-1);
    }
    const DWORD offset_low = static_cast<DWORD>(offset & 0xffffffffu);
    const DWORD offset_high = static_cast<DWORD>(offset >> 32);
    void* view = MapViewOfFile(mapping, FILE_MAP_READ, offset_high, offset_low,
                               length);
    CloseHandle(mapping);
    if (!view) {
        set_errno_from_win();
        return reinterpret_cast<void*>(-1);
    }
    return view;
#else
    void* mapped =
        ::mmap(address, length,
               prot == kProtRead ? PROT_READ : PROT_READ,
               flags == kMapPrivate ? MAP_PRIVATE : MAP_SHARED, fd,
               static_cast<off_t>(offset));
    return mapped;
#endif
}

int munmap(void* address, size_t length) {
#if defined(_WIN32)
    (void)length;
    if (!address)
        return 0;
    if (!UnmapViewOfFile(address)) {
        set_errno_from_win();
        return -1;
    }
    return 0;
#else
    return ::munmap(address, length);
#endif
}

int madvise(void* address, size_t length, int advice) {
#if defined(_WIN32)
    if (!address || length == 0)
        return 0;
    if (advice == kMadvWillneed) {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = address;
        entry.NumberOfBytes = length;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
        return 0;
    }
    // File-backed views have no MADV_DONTNEED equivalent. Leave pages alone.
    (void)advice;
    return 0;
#elif defined(MADV_WILLNEED)
    int posix_advice = MADV_NORMAL;
    if (advice == kMadvWillneed)
        posix_advice = MADV_WILLNEED;
#if defined(MADV_DONTNEED)
    else if (advice == kMadvDontneed)
        posix_advice = MADV_DONTNEED;
#endif
    return ::madvise(address, length, posix_advice);
#else
    (void)address;
    (void)length;
    (void)advice;
    return 0;
#endif
}

int mlock(const void* address, size_t length) {
#if defined(_WIN32)
    if (!address || length == 0)
        return 0;
    if (VirtualLock(const_cast<void*>(address), length))
        return 0;
    SIZE_T min_ws = 0;
    SIZE_T max_ws = 0;
    if (GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws, &max_ws)) {
        const SIZE_T extra = length + (64u << 20);
        SetProcessWorkingSetSize(GetCurrentProcess(), min_ws + extra,
                                 max_ws + extra);
    }
    if (VirtualLock(const_cast<void*>(address), length))
        return 0;
    set_errno_from_win();
    return -1;
#else
    return ::mlock(address, length);
#endif
}

int munlock(const void* address, size_t length) {
#if defined(_WIN32)
    if (!address || length == 0)
        return 0;
    if (!VirtualUnlock(const_cast<void*>(address), length)) {
        set_errno_from_win();
        return -1;
    }
    return 0;
#else
    return ::munlock(address, length);
#endif
}

long page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize > 0 ? static_cast<long>(info.dwPageSize) : 4096L;
#else
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? value : 4096L;
#endif
}

int create_temp(char* path, size_t capacity) {
    if (!path || capacity < 16)
        return -1;
#if defined(_WIN32)
    char directory[MAX_PATH];
    const DWORD n = GetTempPathA(MAX_PATH, directory);
    if (n == 0 || n >= MAX_PATH) {
        set_errno_from_win();
        return -1;
    }
    char generated[MAX_PATH];
    if (!GetTempFileNameA(directory, "mlm", 0, generated)) {
        set_errno_from_win();
        return -1;
    }
    if (std::strlen(generated) + 1 > capacity) {
        errno = ENAMETOOLONG;
        DeleteFileA(generated);
        return -1;
    }
    std::strcpy(path, generated);
    const int fd = _open(path, _O_RDWR | _O_BINARY);
    if (fd < 0)
        DeleteFileA(path);
    return fd;
#else
    static const char kTemplate[] = "/tmp/mollm_pkg_XXXXXX";
    if (capacity < sizeof(kTemplate)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    std::strcpy(path, kTemplate);
    return ::mkstemp(path);
#endif
}

unsigned rand_r(unsigned* seed) {
    unsigned state = seed ? *seed : 42u;
    state = state * 1103515245u + 12345u;
    if (seed)
        *seed = state;
    return (state / 65536u) % 32768u;
}

double peak_rss_mb() {
#if defined(_WIN32)
    struct Counters {
        DWORD cb;
        DWORD page_faults;
        SIZE_T peak_working_set;
        SIZE_T working_set;
        SIZE_T quota_peak_paged;
        SIZE_T quota_paged;
        SIZE_T quota_peak_nonpaged;
        SIZE_T quota_nonpaged;
        SIZE_T pagefile;
        SIZE_T peak_pagefile;
    } counters = {};
    counters.cb = sizeof(counters);
    using Fn = BOOL(WINAPI*)(HANDLE, Counters*, DWORD);
    static const Fn query = reinterpret_cast<Fn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                       "K32GetProcessMemoryInfo"));
    if (!query || !query(GetCurrentProcess(), &counters, sizeof(counters)))
        return 0.0;
    return static_cast<double>(counters.peak_working_set) / (1024.0 * 1024.0);
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0.0;
#if defined(__APPLE__)
    return usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return usage.ru_maxrss / 1024.0;
#endif
#endif
}

}  // namespace mollm::io
