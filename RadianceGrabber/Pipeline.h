#pragma once

#include "Define.h"

namespace RadGrabber
{
	class FrameRequest;
	struct FrameInput;
	class MultiFrameRequest;
	struct MultiFrameInput;

	__host__ void AllocateDeviceFrameRequest(FrameRequest* hostReq, FrameInput** outDeviceInput);
	__host__ void FreeDeviceFrameRequest(FrameInput* deviceInput);
	__host__ void FreeHostFrameRequest(FrameRequest* hreq);
	__host__ size_t StoreFrameRequest(FrameRequest* hostReq, FILE* fp);
	__host__ void LoadFrameRequest(FILE* fp, FrameRequest** reqBuffer, void* (*allocator)(size_t cb));

	__host__ void AllocateDeviceMultiFrameRequest(MultiFrameRequest* hostReq, MultiFrameInput** outDeviceInput);
	__host__ void FreeDeviceMultiFrameRequest(MultiFrameInput* deviceInput);
	__host__ void FreeHostMultiFrameRequest(MultiFrameRequest* hreq);
	__host__ size_t StoreMultiFrameRequest(MultiFrameRequest* hostReq, FILE* fp);
	__host__ void LoadMultiFrameRequest(FILE* fp, MultiFrameRequest** reqBuffer, void* (*allocator)(size_t cb));
}
