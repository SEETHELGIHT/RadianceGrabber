#include <chrono>
#include <ratio>
#include <vector>
#include <cuda.h>
#include <device_functions.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

#include "IterativePathIntegrator.h"

namespace RadGrabber
{
	extern __constant__ struct FrameConstant g_FrameConstant;
	__constant__ struct IterativePathConstant g_IterativePathConstant { };

	__global__ void ResetThreadSegments()
	{
		int threadIndex = getGlobalIdx_3D_3D();
		IterativeThreadSegment& seg = g_IterativePathConstant.threadSegments[threadIndex];

		seg.pidx = threadIndex;
		seg.in.isInitialized = 0;
		seg.in.traversalSegement.initilaized = 0;

		if (threadIndex == 0)
			*g_IterativePathConstant.pixelProcessEndCount = 0;
	}

	__global__ void InitializeKernelIterative()
	{
		int threadIndex = getGlobalIdx_3D_3D();
		IterativeThreadSegment& seg = g_IterativePathConstant.threadSegments[threadIndex];

		if (threadIndex >= g_IterativePathConstant.frameWidth * g_IterativePathConstant.frameHeight)
		{
			seg.pidx = -1;
			return;
		}

		// per thread operation
		curand_init(g_IterativePathConstant.currentTime, threadIndex, 0, g_IterativePathConstant.threadRandStates + threadIndex);

		seg.pidx = threadIndex;
		seg.in = g_IterativePathConstant.pathSegmnetInitValue;

		// per pixel operation
		for (int i = threadIndex; i < g_IterativePathConstant.frameWidth * g_IterativePathConstant.frameHeight; i += g_IterativePathConstant.segCnt)
		{
			GetPixelRay(
				g_IterativePathConstant.cameraProjectionInverseMatrix,
				g_IterativePathConstant.cameraSpaceInverseMatrix,
				i,
				Vector2i(g_IterativePathConstant.frameWidth, g_IterativePathConstant.frameHeight),
				g_IterativePathConstant.pixelSegments[i].rayInWS
			);
			g_IterativePathConstant.pixelSegments[i].atten = ColorRGB(1, 1, 1);
		}

		if (threadIndex == 0)
			*g_IterativePathConstant.pixelProcessEndCount = 0;
	}

#define HITTESTMAX_ITERATION_COUNT 500

	__global__ void HitTestKernel()
	{
		int threadCount = getGlobalIdx_3D_3D();
		IterativeThreadSegment& threadSegment = g_IterativePathConstant.threadSegments[threadCount];
		if (threadSegment.pidx < 0) return;
		int remainIterateCount = HITTESTMAX_ITERATION_COUNT;

		if (!threadSegment.in.isInitialized)
		{
			threadSegment.in.isInitialized = 1;
			threadSegment.in.traversalSegement.isLowerTransform = 0;
			threadSegment.in.traversalSegement.findPrimitive = 0;
		}

	HITTEST_PIXEL_INITIALIZATION:

		IterativePixelSegment& pixelSegment = g_IterativePathConstant.pixelSegments[threadSegment.pidx];

	HITTEST_GEOMETRY_INTERSECTION:

		if (
			g_IterativePathConstant.agg->IterativeIntersect(
				pixelSegment.rayInWS, threadSegment.rayInMS, pixelSegment.isect, threadCount,
				remainIterateCount, threadSegment.in.traversalSegement
			)
			)
		{
			if (pixelSegment.isect.isHit &&
				pixelSegment.isect.isGeometry &&
				threadSegment.in.remainBounces - 1 >= 0)
			{
				// continue surface scattering
				threadSegment.in.remainBounces--;
				goto HITTEST_GEOMETRY_INTERSECTION;
			}
			else
			{
				if (threadSegment.pidx + g_IterativePathConstant.segCnt < g_IterativePathConstant.frameWidth * g_IterativePathConstant.frameHeight)
				{
					threadSegment.pidx += g_IterativePathConstant.segCnt;
					goto HITTEST_PIXEL_INITIALIZATION;
				}
				else
				{
					atomicAdd(g_IterativePathConstant.pixelProcessEndCount, 1);
					threadSegment.pidx = -1;
					return;
				}
			}
		}
		else
		{
			return;
		}
	}

#define VISIBLELIGHTMAX_ITERATION_COUNT 500

	__global__ void VisibleLightTestKernel()
	{
		int threadCount = getGlobalIdx_3D_3D();
		IterativeThreadSegment& threadSegment = g_IterativePathConstant.threadSegments[threadCount];
		if (threadSegment.pidx < 0) return;
		int remainIterateCount = VISIBLELIGHTMAX_ITERATION_COUNT;

		const FrameMutableInput* min = g_IterativePathConstant.in->GetMutable(g_IterativePathConstant.mutableIndex);;
		LightChunk* chunks = min->lightBuffer;
		int lightCount = min->lightBufferLen;

		if (!threadSegment.in.isInitialized)
		{
			threadSegment.in.isInitialized = 1;
			threadSegment.lightIndex = 0;
		}

	LIGHTTEST_PIXEL_INITIALIZATION:

		IterativePixelSegment& pixelSegment = g_IterativePathConstant.pixelSegments[threadSegment.pidx];

		for (; threadSegment.lightIndex < lightCount; threadSegment.lightIndex++)
		{
			const LightChunk& light = chunks[threadSegment.lightIndex];
			switch (light.type)
			{
			case eUnityLightType::Directional:
			{
				if (
					g_IterativePathConstant.agg->IterativeIntersectOnly(
						Ray(pixelSegment.isect.position, -light.forward),
						threadSegment.rayInMS,
						threadCount,
						remainIterateCount,
						threadSegment.in.traversalSegement
					)
					)
				{
					pixelSegment.infLightColor = pixelSegment.infLightColor + fmin(fmax(Dot(pixelSegment.isect.normal, -light.forward), 0.f), 1.f) * light.color * light.intensity;
				}
				else
					return;

			}
			break;
			case eUnityLightType::Point:
			{
				if (
					g_IterativePathConstant.agg->IterativeIntersectOnly(
						Ray(pixelSegment.isect.position, (light.position - pixelSegment.isect.position)),
						threadSegment.rayInMS,
						threadCount,
						remainIterateCount,
						threadSegment.in.traversalSegement
					)
					)
				{
					float sqd = (light.position - pixelSegment.isect.position).sqrMagnitude();
					pixelSegment.infLightColor = pixelSegment.infLightColor + light.color / sqd * light.intensity * (sqd <= light.range);
				}
				else
					return;
			}
			break;
			case eUnityLightType::Spot:
			{
				if (
					g_IterativePathConstant.agg->IterativeIntersectOnly(
						Ray(pixelSegment.isect.position, (light.position - pixelSegment.isect.position)),
						threadSegment.rayInMS,
						threadCount,
						remainIterateCount,
						threadSegment.in.traversalSegement
					)
					)
				{
					Vector3f dir = light.position - pixelSegment.isect.position;
					float d = dir.magnitude();
					dir /= d;

					if (d <= light.range && (Dot(dir, light.forward)) < cos(light.angle))
						pixelSegment.infLightColor = pixelSegment.infLightColor + light.color / d / d * light.intensity;
				}
				else
					return;
			}
			break;
			}
		}

		if (threadSegment.pidx + g_IterativePathConstant.segCnt < g_IterativePathConstant.frameWidth * g_IterativePathConstant.frameHeight)
		{
			threadSegment.pidx += g_IterativePathConstant.segCnt;
			goto LIGHTTEST_PIXEL_INITIALIZATION;
		}
		else
		{
			atomicAdd(g_IterativePathConstant.pixelProcessEndCount, 1);
			threadSegment.pidx = -1;
			return;
		}
	}

	__global__ void ColorCalculationKernel()
	{
		const FrameMutableInput* min = g_IterativePathConstant.in->GetMutable(g_IterativePathConstant.mutableIndex);;
		int lastSamplingCount = *g_IterativePathConstant.samplingCount;

		LightChunk* chunks = min->lightBuffer;
		int skyboxIndex = min->cameraBuffer[min->selectedCameraIndex].skyboxIndex;

		for (int i = getGlobalIdx_3D_3D(); i < g_IterativePathConstant.frameWidth * g_IterativePathConstant.frameHeight; i += g_IterativePathConstant.segCnt)
		{
			IterativePixelSegment& pixelSegment = g_IterativePathConstant.pixelSegments[i];
			Vector3f c0 = Vector3f(0.f, 0.f, 0.f), c1 = Vector3f(0.f, 0.f, 0.f);

			if (pixelSegment.isect.isHit)
				if (!pixelSegment.isect.isGeometry)
					c0 = chunks[pixelSegment.isect.itemIndex].color;

			if (skyboxIndex >= 0)
				min->skyboxMaterialBuffer[skyboxIndex].Sample(pixelSegment.rayInWS, g_IterativePathConstant.in->GetImmutable()->textureBuffer, c1);

			g_IterativePathConstant.cb[i] =
				g_IterativePathConstant.cb[i] * (float)(lastSamplingCount - 1) / lastSamplingCount +
				pixelSegment.atten * (ColorRGB)(c0 + c1 + pixelSegment.infLightColor) / (float)lastSamplingCount;
		}
	}

	__host__ IterativePathIntegrator::IterativePathIntegrator(const RequestOption& opt, const OptimalLaunchParam& param) : reserveCancel(false)
	{
		mSegmentCount = param.blockCountInGrid.x * param.threadCountinBlock.x;
		mDeviceSegments = (IterativeThreadSegment*)MAllocDevice(sizeof(IterativeThreadSegment) * mSegmentCount);
		gpuErrchk(cudaMemset(mDeviceSegments, 0, sizeof(IterativeThreadSegment) * mSegmentCount));
		mHostSegments = (IterativeThreadSegment*)MAllocHost(sizeof(IterativeThreadSegment) * mSegmentCount);
		gpuErrchk(cudaMemset(mHostSegments, 0, sizeof(IterativeThreadSegment) * mSegmentCount));
		mSegmentInitFlags = (int*)MAllocDevice(sizeof(curandState) * (int)ceil((float)mSegmentCount / 32.f));
		gpuErrchk(cudaMemset(mHostSegments, 0, sizeof(IterativeThreadSegment) * (int)ceil((float)mSegmentCount / 32.f)));
		mDevicePixelSegments = (IterativePixelSegment*)MAllocDevice(sizeof(IterativePixelSegment) * opt.resultImageResolution.x * opt.resultImageResolution.y);
		gpuErrchk(cudaMemset(mDevicePixelSegments, 0, sizeof(IterativePixelSegment) * opt.resultImageResolution.x * opt.resultImageResolution.y));
		mRandStates = (curandState*)MAllocDevice(sizeof(curandState) * mSegmentCount);

		Log("Segment Count::%d\n", mSegmentCount);
	}

	__host__ IterativePathIntegrator::~IterativePathIntegrator()
	{
		SAFE_HOST_DELETE(mHostSegments);
		SAFE_DEVICE_DELETE(mDeviceSegments);
		SAFE_DEVICE_DELETE(mSegmentInitFlags);
		SAFE_DEVICE_DELETE(mDevicePixelSegments);
		SAFE_DEVICE_DELETE(mRandStates);
	}

	__host__ void IterativePathIntegrator::RenderIncremental(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
		int maxSharedMemorySize = 0;
		std::chrono::system_clock::time_point
			lastUpdateTime = std::chrono::system_clock::now();
		cudaStream_t kernelExecuteStream = (cudaStream_t)0;
		CameraChunk* dc = (CameraChunk*)MAllocDevice(sizeof(CameraChunk));

#ifndef RELEASE
		AccumulatedTimeProfiler
			initProfile("PT init profile"),
			kernelProfile("PT kernel profile"),
			updateFuncPRofile("PT update function call profile"),
			ptProfile("PT whoe profile");
#endif
		ptProfile.StartProfile();

		IterativePathConstant initParam;

		initParam.frameWidth = opt.resultImageResolution.x;
		initParam.frameHeight = opt.resultImageResolution.y;
		initParam.cb = (ColorRGBA*)target.GetDeviceColorBuffer();
		initParam.in = in.device;

		initParam.segCnt = mSegmentCount;
		initParam.threadSegments = mDeviceSegments;
		initParam.threadRandStates = mRandStates;
		initParam.threadSegmentInitFlag = mSegmentInitFlags;
		initParam.pixelSegments = mDevicePixelSegments;

		initParam.pixelProcessEndCount = (int*)MAllocDevice(sizeof(int) * 1);
		initParam.samplingCount = (int*)MAllocDevice(sizeof(int) * 1);

		initParam.pathSegmnetInitValue.flag = 0.f;
		initParam.pathSegmnetInitValue.remainBounces = opt.maxDepth;

		InitTraversalSegment(initParam.pathSegmnetInitValue.traversalSegement);

		gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1));

		for (int i = 0; i < in.host->GetCount(); i++)
		{
			const FrameMutableInput* min = in.host->GetMutable(i);

			initParam.agg = getDeviceScene(i);
			initParam.cameraProjectionInverseMatrix = min->cameraBuffer[min->selectedCameraIndex].projectionInverseMatrix;
			initParam.cameraSpaceInverseMatrix = min->cameraBuffer[min->selectedCameraIndex].cameraInverseMatrix;

			gpuErrchk(cudaMemcpyToSymbol(g_IterativePathConstant, &initParam, sizeof(IterativePathConstant)));

			Log("Mutable Index::%d, Start\n", i);

			if (reserveCancel)
			{
				reserveCancel = false;
				return;
			}

#ifndef RELEASE
			initProfile.StartProfile();
#endif

			InitializeKernelIterative << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
			gpuErrchk(cudaGetLastError());

#ifndef RELEASE
			initProfile.EndProfileAndAccumulate();
#endif
			int samplingCount = 0;

			while (samplingCount++ < opt.maxSamplingCount)
			{
				gpuErrchk(cudaMemcpy(g_IterativePathConstant.samplingCount, &samplingCount, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
#ifndef RELEASE
				kernelProfile.StartProfile();
#endif
				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				int cnt;
				cnt = 0;
				do
				{
					HitTestKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
					gpuErrchk(cudaGetLastError());
					gpuErrchk(cudaMemcpy(&cnt, g_IterativePathConstant.pixelProcessEndCount, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				} while (cnt < initParam.frameWidth * initParam.frameHeight);

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				ResetThreadSegments << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
				gpuErrchk(cudaGetLastError());

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}
				cnt = 0;
				do
				{
					VisibleLightTestKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
					gpuErrchk(cudaGetLastError());
					gpuErrchk(cudaMemcpy(&cnt, g_IterativePathConstant.pixelProcessEndCount, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				} while (cnt < initParam.frameWidth * initParam.frameHeight);

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				ColorCalculationKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
				gpuErrchk(cudaGetLastError());

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}
#ifndef RELEASE
				kernelProfile.EndProfileAndAccumulate();
#endif			
			}

			Log("Mutable Index::%d, End\n", i);
		}


		{
			updateFuncPRofile.StartProfile();
			target.UploadDeviceToHost();
			if (opt.updateFunc) opt.updateFunc(opt.maxSamplingCount);
			lastUpdateTime = std::chrono::system_clock::now();
			updateFuncPRofile.EndProfileAndAccumulate();
		}

		SAFE_DEVICE_DELETE(initParam.pixelProcessEndCount);
		SAFE_DEVICE_DELETE(initParam.samplingCount);

		SAFE_DEVICE_DELETE(dc);

		ptProfile.EndProfileAndAccumulate();

#ifndef RELEASE
		initProfile.Print();
		kernelProfile.Print();
		updateFuncPRofile.Print();
		ptProfile.Print();
#endif
	}

	__host__ void IterativePathIntegrator::RenderStraight(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
	}

	__host__ void IterativePathIntegrator::ReserveCancel()
	{
		reserveCancel = true;
	}

	__host__ bool IterativePathIntegrator::IsCancel()
	{
		return !reserveCancel;
	}
}