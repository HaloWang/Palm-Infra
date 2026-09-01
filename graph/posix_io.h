#pragma once

// Portable file, mmap, and page-lock helpers.
// POSIX builds call the native APIs. Windows maps them onto CRT fds plus
// CreateFileMapping / ReadFile / VirtualLock. MADV_DONTNEED is best-effort
// on Windows (file-backed pages cannot be discarded the same way).

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#include <io.h>
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
using ssize_t = std::ptrdiff_t;
#endif
struct iovec {
    void* iov_base;
    size_t iov_len;
};
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace mollm::io {

constexpr int kProtRead = 1;
constexpr int kMapPrivate = 2;
constexpr int kMadvWillneed = 3;
constexpr int kMadvDontneed = 4;

int open_read(const char* path);
int close(int fd);
int write(int fd, const void* data, size_t length);
int file_size(int fd, uint64_t* size);
ssize_t pread(int fd, void* buffer, size_t length, uint64_t offset);
ssize_t preadv(int fd, const iovec* vectors, int count, uint64_t offset);
void* mmap(void* address, size_t length, int prot, int flags, int fd,
           uint64_t offset);
int munmap(void* address, size_t length);
int madvise(void* address, size_t length, int advice);
int mlock(const void* address, size_t length);
int munlock(const void* address, size_t length);
long page_size();
int create_temp(char* path, size_t capacity);
unsigned rand_r(unsigned* seed);
double peak_rss_mb();

}  // namespace mollm::io
