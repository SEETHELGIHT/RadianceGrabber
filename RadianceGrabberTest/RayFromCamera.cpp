#include "Marshal.cuh"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		__host__ Vector2i GetImageIndex(int startIndex, int pixelIndex, const Vector2i& texRes)
		{
			return Vector2i(pixelIndex % texRes.x, (int)pixelIndex / texRes.x);
		}

		int RayFromCamera()
		{
			FILE* fp = nullptr;
			fopen_s(&fp, "./frq.framerequest", "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			// Generate Ray
			const FrameMutableInput* in = req->input.GetMutable(0);
			CameraChunk& c = in->cameraBuffer[in->selectedCameraIndex];
			Ray r;


			printf("%.2f, %.2f, %.2f\n", c.position.x, c.position.y, c.position.z);
			printf("%.2f, %.2f, %.2f\n", r.direction.x, r.direction.y, r.direction.z);

			return 0;
		}
		
	}
}