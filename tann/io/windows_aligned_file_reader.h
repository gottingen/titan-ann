// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "tann/io/aligned_file_reader.h"
#include "tann/tsl/robin_map.h"
#include "tann/common/utils.h"
#include "turbo/platform/port.h"

class WindowsAlignedFileReader : public AlignedFileReader
{
  private:
#ifdef UNICODE
    std::wstring m_filename;
#else
    std::string m_filename;
#endif

  protected:
    // virtual IOContext createContext();

  public:
    TURBO_DLL WindowsAlignedFileReader(){};
    TURBO_DLL virtual ~WindowsAlignedFileReader(){};

    // Open & close ops
    // Blocking calls
    TURBO_DLL virtual void open(const std::string &fname) override;
    TURBO_DLL virtual void close() override;

    TURBO_DLL virtual void register_thread() override;
    TURBO_DLL virtual void deregister_thread() override
    {
        // TODO: Needs implementation.
    }
    TURBO_DLL virtual void deregister_all_threads() override
    {
        // TODO: Needs implementation.
    }
    TURBO_DLL virtual IOContext &get_ctx() override;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call for the calling thread, but can thread-safe
    TURBO_DLL virtual void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async) override;
};
#endif // USE_BING_INFRA
#endif //_WINDOWS
