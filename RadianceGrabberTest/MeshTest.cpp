#include <cuda_runtime_api.h>
#include <cstdio>
#include <chrono>

#include "Util.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "integrator.h"
#include "Marshal.cuh"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Unity/RenderAPI.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		__global__ void MeshTest2(IMultipleInput* in)
		{
			auto* imin = in->GetImmutable();
			auto* min = in->GetMutable(0);

			for (int i = 0; i < min->meshRendererBufferLen; i++)
			{
				auto& mr = min->meshRendererBuffer[i];
				auto& mc = imin->meshBuffer[mr.meshRefIndex];

				for (int j = 0; j < mc.submeshCount; j++)
				{
					Bounds b = imin->meshBuffer[mr.meshRefIndex].submeshArrayPtr[j].bounds;

					b.center = Vector3f::Zero();
				}
			}
		}

		int MeshTest()
		{
			FILE* fp = fopen("./frq.framerequest", "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			FrameInput* din = nullptr;
			AllocateDeviceMem(req, &din);
			
			MeshTest2 << <1, 1 >> > (din);

			return 0;
		}
	}
}