#pragma once

#include "Define.h"

namespace RadGrabber
{
	struct UnityFrameRequest;
	struct UnityFrameInput;

	__host__ void AllocateDeviceMem(UnityFrameRequest* hostReq, UnityFrameInput** outDeviceInput);
	__host__ void FreeDeviceMem(UnityFrameInput* deviceInput);
}