#include <cuda_runtime_api.h>
#include <cstdio>
#include <chrono>
#include <direct.h>

#define RADIANCEGRABBER_REMOVE_LOG
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
		std::chrono::system_clock::time_point sphereLastUpdateTime = std::chrono::system_clock::now();
		SimpleColorTarget* sphereTarget;
		const char* sphereFileDirAndName = "./sphere.ppm";

		void UpdateSphereImage(int cnt)
		{
			printf("cnt::%d, %ld\n", cnt, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sphereLastUpdateTime).count());
			FILE* file = fopen(sphereFileDirAndName, "wt");
			sphereTarget->WritePPM(file);
			fclose(file);
			sphereLastUpdateTime = std::chrono::system_clock::now();
		}
		int SphereTest()
		{
			RequestOption opt;
			opt.maxSamplingCount = 50;
			opt.resultImageResolution = Vector2i(128, 128);
			opt.updateFunc = UpdateSphereImage;

			sphereTarget = new SimpleColorTarget(opt.resultImageResolution.x, opt.resultImageResolution.y);

			PathIntegrator* integrator = new PathIntegrator(opt);

			FILE* fp = fopen("./SimpleScene.framerequest", "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			Sphere* s = Sphere::GetSphereDevice(Vector3f(0, 0.5f, 0), 1);
			req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].position = Vector3f(9.206837, 5.451627, 3.552553);
			req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].forward = Vector3f(-0.5, -0.6, -0.6);
			req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].quaternion = Quaternion(-0.1, 0.9, -0.3, -0.3);

			FrameInput* din = nullptr;
			AllocateDeviceMem(req, &din);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);
			integrator->Render(*s, req->input, *din, opt, *sphereTarget, param);

			gpuErrchk(cudaDeviceReset());

			char cCurrentPath[2024];
			_getcwd(cCurrentPath, sizeof(cCurrentPath));
			strcat_s(cCurrentPath, sizeof(cCurrentPath), sphereFileDirAndName + 1);
			strcat_s(cCurrentPath, sizeof(cCurrentPath), " & ");
			system(cCurrentPath);

			return 0;
		}
	}
}
