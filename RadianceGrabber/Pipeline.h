#pragma once

#include "Define.h"

namespace RadGrabber
{
	struct FrameRequest;
	struct FrameInput;

	__host__ void AllocateDeviceMem(FrameRequest* hostReq, FrameInput** outDeviceInput);
	__host__ void FreeDeviceMem(FrameInput* deviceInput);
}