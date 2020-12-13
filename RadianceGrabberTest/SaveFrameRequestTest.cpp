#include <cuda_runtime_api.h>
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <chrono>

#include "integrator.h"
#include "Marshal.cuh"
#include "Util.h"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Pipeline.h"
#include "DLLEntry.h"

namespace RadGrabber
{
	namespace Test
	{
		int SaveFrameRequestTest()
		{
			FILE* fp = nullptr;
			fopen_s(&fp, "./frq.framerequest", "rb");
			FrameRequest* frm = new FrameRequest();
			LoadFrameRequest(fp, &frm, malloc);
			fclose(fp);

			FrameRequestMarshal* frqMarshal = new FrameRequestMarshal();
			frqMarshal->opt = frm->opt;
			frqMarshal->input = frm->input.in;
			frqMarshal->output = frm->output;

			::SaveSingleFrameRequest("SaveFrameRequestTest", frqMarshal);

			return 0;
		}
	}
}
