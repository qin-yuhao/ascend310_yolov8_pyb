#pragma once
#include <iostream>
#include "acl/acl.h"
#include "type_api.h"

using namespace std;

class basic
{
private:
    /* data */
    uint32_t g_modelId_;
    int32_t g_deviceId_;  // Device ID, default is 0

    
    aclmdlDesc *g_modelDesc_;
    aclrtContext g_context_;
    aclrtStream g_stream_;
    

    
    
    const char* omModelPath;

public:
    basic(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight);
    ~basic();
    
    void*  g_imageDataBuf_;      // Model input data cache
    uint32_t g_imageDataSize_; // Model input data size
    aclrtRunMode g_runMode_;
    int outputSize;

public:
    Result Init_atlas();
    Result CreateInput();
    Result CreateOutput();
    Result inference();
    void* GetInferenceOutputItem(uint32_t idx);
    void* CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize);

    
    void DestroyResource();
    void DestroyDesc();
    void DestroyInput();
    void DestroyOutput();
    aclmdlDataset *g_input_;
    aclmdlDataset *g_output_;
};

