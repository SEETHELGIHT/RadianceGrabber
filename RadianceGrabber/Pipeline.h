#pragma once

#include "Define.h"

namespace RadGrabber
{
	struct FrameRequest;
	struct FrameInput;

	__host__ void AllocateDeviceMem(FrameRequest* hostReq, FrameInput** outDeviceInput);
	__host__ void FreeDeviceMem(FrameInput* deviceInput);
	
	__host__ void FreeHostMem(FrameRequest* hreq);

	__host__ size_t StoreFrameRequest(FrameRequest* hostReq, FILE* fp);
	__host__ void LoadFrameRequest(FILE* fp, FrameRequest** reqBuffer, void* (*allocator)(size_t cb));
}