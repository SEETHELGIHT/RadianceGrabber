#include "PathIntegrator.h"

#include <chrono>
#include <thread>
#include <ratio>
#include <vector>
#include <cuda.h>
#include <device_functions.h>
#include <cooperative_groups.h>
#include <ctime>

using namespace cooperative_groups;

#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{
	__constant__ struct PathConstant g_PathConstants { };

	__host__ PathIntegrator::PathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution, bool incrementalMode) : reserveCancel(false), mPixelContent(PathIntegratorPixelContent::Luminance), mIncrementalMode(incrementalMode)
	{
		mSegmentCount = (param.blockCountInGrid.x * param.blockCountInGrid.y * param.blockCountInGrid.z) * (param.threadCountinBlock.x * param.threadCountinBlock.y * param.threadCountinBlock.z);
		mDeviceSegments = (PathThreadSegment*)MAllocDevice(sizeof(PathThreadSegment) * mSegmentCount);
		gpuErrchk(cudaMemset(mDeviceSegments, 0, sizeof(PathThreadSegment) * mSegmentCount));
		mHostSegments = (PathThreadSegment*)MAllocHost(sizeof(PathThreadSegment) * mSegmentCount);
		memset(mHostSegments, 0, sizeof(PathThreadSegment) * mSegmentCount);
		//gpuErrchk(cudaMemset(mHostSegments, 0, sizeof(PathThreadSegment) * mSegmentCount));
		mRandStates = (curandState*)MAllocDevice(sizeof(curandState) * mSegmentCount);

		mPixelSegments = (PathPixelSegment*)MAllocDevice(sizeof(PathPixelSegment) * resultImageResolution.x * resultImageResolution.y);

		Log("Segment Count::%d\n", mSegmentCount);
	}
	__host__ PathIntegrator::PathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution, PathIntegratorPixelContent pixelContent, bool incrementalMode) : reserveCancel(false), mPixelContent(pixelContent), mIncrementalMode(incrementalMode)
	{
		mSegmentCount = (param.blockCountInGrid.x * param.blockCountInGrid.y * param.blockCountInGrid.z) * (param.threadCountinBlock.x * param.threadCountinBlock.y * param.threadCountinBlock.z);
		mDeviceSegments = (PathThreadSegment*)MAllocDevice(sizeof(PathThreadSegment) * mSegmentCount);
		gpuErrchk(cudaMemset(mDeviceSegments, 0, sizeof(PathThreadSegment) * mSegmentCount));
		mHostSegments = (PathThreadSegment*)MAllocHost(sizeof(PathThreadSegment) * mSegmentCount);
		memset(mHostSegments, 0, sizeof(PathThreadSegment) * mSegmentCount);
		//gpuErrchk(cudaMemset(mHostSegments, 0, sizeof(PathThreadSegment) * mSegmentCount));
		mRandStates = (curandState*)MAllocDevice(sizeof(curandState) * mSegmentCount);

		mPixelSegments = (PathPixelSegment*)MAllocDevice(sizeof(PathPixelSegment) * resultImageResolution.x * resultImageResolution.y);

		Log("Segment Count::%d\n", mSegmentCount);
	}

	__host__ PathIntegrator::~PathIntegrator()
	{
		SAFE_DEVICE_DELETE(mDeviceSegments);
		SAFE_HOST_DELETE(mHostSegments);
		SAFE_DEVICE_DELETE(mRandStates);
		SAFE_DEVICE_DELETE(mPixelSegments);
	}

	__global__ void SetDeviceInitParam(IMultipleInput* din, PathConstant* c)
	{
		c->mutableIn = *din->GetMutable(c->mutableIndex);
		c->immutable = *din->GetImmutable();
	}

	__host__ void PathIntegrator::RenderStraight(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
		int	*devicePixelEndCount = (int*)MAllocDevice(sizeof(int) * 2),
			*nextPIxelCount = (int*)MAllocDevice(sizeof(int)),
			maxSharedMemorySize = 0;
		std::chrono::system_clock::time_point
			lastUpdateTime = std::chrono::system_clock::now();
		cudaStream_t kernelExecuteStream = (cudaStream_t)0;
		CameraChunk* dc = (CameraChunk*)MAllocDevice(sizeof(CameraChunk));

#ifndef RELEASE
		AccumulatedTimeProfiler
			initProfile("PT init profile"),
			kernelProfile("PT kernel profile"),
			pixelProfile("PT pixel profile"),
			updateFuncPRofile("PT update function call profile"),
			ptProfile("PT whoe profile");
#endif
		ptProfile.StartProfile();

		gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1));

		PathConstant hostInitParam, *deviceParamInitParam = (PathConstant*)MAllocDevice(sizeof(PathConstant));
		hostInitParam.SetConstantValues(false, opt.maxSamplingCount, opt.maxDepth, opt.resultImageResolution, mSegmentCount, lastUpdateTime.time_since_epoch().count(), opt.threadIterateCount);

		hostInitParam.cb = (ColorRGBA*)target.GetDeviceColorBuffer();
		hostInitParam.in = in.device;
		hostInitParam.segs = mDeviceSegments;
		hostInitParam.pixels = mPixelSegments;
		hostInitParam.randStates = mRandStates;
		hostInitParam.devicePixelEndCount = devicePixelEndCount;
		hostInitParam.nextPixelCount = nextPIxelCount;
		hostInitParam.pixelContent = mPixelContent;
		hostInitParam.incrementalMode = mIncrementalMode;

		for (int i = in.host->GetStartIndex(); i < in.host->GetCount(); i++)
		{
			const FrameMutableInput* min = in.host->GetMutable(i);

			hostInitParam.agg = getDeviceScene(i);
			hostInitParam.SetFrameConstantValues(i, *min);

			gpuErrchk(cudaMemcpy(deviceParamInitParam, &hostInitParam, sizeof(PathConstant), cudaMemcpyKind::cudaMemcpyHostToDevice));
			SetDeviceInitParam << <1, 1 >> > (in.device, deviceParamInitParam);
			gpuErrchk(cudaMemcpyToSymbol(g_PathConstants, deviceParamInitParam, sizeof(PathConstant)));

			Log("Mutable Index::%d, Start\n", i);

			if (reserveCancel)
			{
				reserveCancel = false;
				goto ENDOF_RENDERSTRAIGHT;
			}

			gpuErrchk(cudaMemset(hostInitParam.cb, 0, sizeof(ColorRGBA) * opt.resultImageResolution.x * opt.resultImageResolution.y));

#ifndef RELEASE
			initProfile.StartProfile();
#endif

			InitializeKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();

#ifndef RELEASE
			initProfile.EndProfileAndAccumulate();
#endif

			if (reserveCancel)
			{
				reserveCancel = false;
				goto ENDOF_RENDERSTRAIGHT;
			}

			//gpuErrchk(cudaMemset(nextPIxelCount, 0, sizeof(int)));
			gpuErrchk(cudaMemcpy(nextPIxelCount, &hostInitParam.segCnt, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
			while (true)
			{
#ifndef RELEASE
				kernelProfile.StartProfile();
#endif

				IntersectKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
				gpuErrchk(cudaGetLastError());

#ifndef RELEASE
				kernelProfile.EndProfileAndAccumulate();
#endif
				int cnt;
				gpuErrchk(cudaMemcpy(&cnt, nextPIxelCount, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

				if (cnt >= opt.resultImageResolution.x * opt.resultImageResolution.y + hostInitParam.segCnt)
					break;

				if (reserveCancel)
				{
					reserveCancel = false;
					goto ENDOF_RENDERSTRAIGHT;
				}
			}

			//gpuErrchk(cudaMemcpy(nextPIxelCount, &hostInitParam.segCnt, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
			//ColorKernel<< <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
			//gpuErrchk(cudaGetLastError());

			if (reserveCancel)
			{
				reserveCancel = false;
				goto ENDOF_RENDERSTRAIGHT;
			}

#ifndef RELEASE
			updateFuncPRofile.StartProfile();
			{
#endif
				gpuErrchk(cudaDeviceSynchronize());
				target.UploadDeviceToHost();
				if (opt.updateFunc != nullptr)
					opt.updateFunc(i);
				if (opt.updateFrameFunc != nullptr)
					opt.updateFrameFunc(i, target.GetHostColorBuffer());
#ifndef RELEASE
			}
			updateFuncPRofile.EndProfileAndAccumulate();
#endif

			Log("Mutable Index::%d, End\n", i);
		}

ENDOF_RENDERSTRAIGHT:
		gpuErrchk(cudaDeviceSynchronize());

		SAFE_DEVICE_DELETE(deviceParamInitParam);
		SAFE_DEVICE_DELETE(devicePixelEndCount);
		SAFE_DEVICE_DELETE(dc);

		ptProfile.EndProfileAndAccumulate();

#ifndef RELEASE
		initProfile.Print();
		kernelProfile.Print();
		pixelProfile.Print();
		ptProfile.Print();
		updateFuncPRofile.Print();
#endif
	}

	__host__ void PathIntegrator::RenderIncremental(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
		printf("RenderIncremental\n");

		int samplingCount = 0,
			prevDrawSamplingCount = samplingCount,
			maxSharedMemorySize = 0;
		const int
			drawSamplingCount = 4,
			updateMilliSecond = 500;

		int endOfPixelSegmentCount = 0;
		std::chrono::system_clock::time_point
			lastUpdateTime = std::chrono::system_clock::now();
		cudaStream_t kernelExecuteStream = (cudaStream_t)0;
		CameraChunk* dc = (CameraChunk*)MAllocDevice(sizeof(CameraChunk));

		int idlePixelSegmentCount[2] = { mSegmentCount - 1, 0 };
		int* devicePixelEndCount = (int*)MAllocDevice(sizeof(int) * 2);
		int *nextPIxelCount = (int*)MAllocDevice(sizeof(int));

		gpuErrchk(cudaMemcpy(devicePixelEndCount, idlePixelSegmentCount, sizeof(int) * 2, cudaMemcpyKind::cudaMemcpyHostToDevice));

#ifndef RELEASE
		AccumulatedTimeProfiler
			initProfile("PT init profile"),
			kernelProfile("PT kernel profile"),
			pixelProfile("PT pixel profile"),
			pixelHostProfile("PT host pixel profile"),
			pixelDeviceProfile("PT device pixel profile"),
			updateFuncPRofile("PT update function call profile"),
			ptProfile("PT whoe profile");
#endif
		ptProfile.StartProfile();

		PathConstant hostInitParam, *deviceParamInitParam = (PathConstant*)MAllocDevice(sizeof(PathConstant));
		hostInitParam.SetConstantValues(true, 1, opt.maxDepth, opt.resultImageResolution, mSegmentCount, lastUpdateTime.time_since_epoch().count(), opt.threadIterateCount);

		hostInitParam.cb = (ColorRGBA*)target.GetDeviceColorBuffer();
		hostInitParam.in = in.device;
		hostInitParam.segs = mDeviceSegments;
		hostInitParam.pixels = mPixelSegments;
		hostInitParam.randStates = mRandStates;
		hostInitParam.devicePixelEndCount = devicePixelEndCount;
		hostInitParam.nextPixelCount = nextPIxelCount;
		hostInitParam.pixelContent = mPixelContent;
		hostInitParam.incrementalMode = mIncrementalMode;
		

		gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1));

		for (int i = in.host->GetStartIndex(); i < in.host->GetCount(); i++)
		{
			const FrameMutableInput* min = in.host->GetMutable(i);

			hostInitParam.currentTime = std::time(nullptr);
			hostInitParam.agg = getDeviceScene(i);
			hostInitParam.SetFrameConstantValues(i, *min);

			gpuErrchk(cudaMemcpy(deviceParamInitParam, &hostInitParam, sizeof(PathConstant), cudaMemcpyKind::cudaMemcpyHostToDevice));
			SetDeviceInitParam << <1, 1 >> > (in.device, deviceParamInitParam);
			gpuErrchk(cudaMemcpyToSymbol(g_PathConstants, deviceParamInitParam, sizeof(PathConstant)));

			gpuErrchk(cudaMemset(hostInitParam.cb, 0, sizeof(ColorRGBA) * opt.resultImageResolution.x * opt.resultImageResolution.y));

			Log("Mutable Index::%d, Start\n", i);

			while (samplingCount < opt.maxSamplingCount)
			{
				TimeProfiler s("Sampling Once");

				Log("Sampling::%d, Start\n", samplingCount);

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

#ifndef RELEASE
				initProfile.StartProfile();
#endif

				InitializeKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
				gpuErrchk(cudaMemcpy(mHostSegments, mDeviceSegments, sizeof(PathThreadSegment) * mSegmentCount, cudaMemcpyKind::cudaMemcpyDeviceToHost));

#ifndef RELEASE
				initProfile.EndProfileAndAccumulate();
#endif

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				int cnt;
				int loop = 0;
				gpuErrchk(cudaMemcpy(&cnt, nextPIxelCount, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				printf("cnt :: %d\n", cnt);

				//gpuErrchk(cudaMemset(nextPIxelCount, 0, sizeof(int)));
				gpuErrchk(cudaMemcpy(nextPIxelCount, &hostInitParam.segCnt, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));

				while (true)
				{
#ifndef RELEASE
					kernelProfile.StartProfile();
#endif
					IntersectKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
					gpuErrchk(cudaGetLastError());

#ifndef RELEASE
					kernelProfile.EndProfileAndAccumulate();
					pixelProfile.StartProfile();
#endif
					int cnt;
					gpuErrchk(cudaMemcpy(&cnt, nextPIxelCount, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
					printf("loop :: %d, cnt :: %d\n", loop, cnt);

#ifndef RELEASE
					pixelProfile.StartProfile();
#endif

					target.UploadDeviceToHost();

					if (cnt >= opt.resultImageResolution.x * opt.resultImageResolution.y + hostInitParam.segCnt)
					{
						break;
					}

					if (reserveCancel)
					{
						reserveCancel = false;
						return;
					}

					loop++;
				}

				//gpuErrchk(cudaMemcpy(nextPIxelCount, &hostInitParam.segCnt, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
				//ColorKernel << <param.blockCountInGrid, param.threadCountinBlock, maxSharedMemorySize, kernelExecuteStream >> > ();
				//gpuErrchk(cudaGetLastError());

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				updateFuncPRofile.StartProfile();
				{
					target.UploadDeviceToHost();
					if (opt.updateFunc != nullptr) opt.updateFunc(samplingCount);
				}
				updateFuncPRofile.EndProfileAndAccumulate();

				Log("Sampling::%d, End\n", samplingCount);

				s.Print();

				samplingCount++;
			}

			if (opt.updateFrameFunc != nullptr)
				opt.updateFrameFunc(i, target.GetHostColorBuffer());

			Log("Mutable Index::%d, End\n", i);
		}

		gpuErrchk(cudaDeviceSynchronize());
		
		SAFE_DEVICE_DELETE(deviceParamInitParam);
		SAFE_DEVICE_DELETE(devicePixelEndCount);
		SAFE_DEVICE_DELETE(nextPIxelCount);
		SAFE_DEVICE_DELETE(dc);

		ptProfile.EndProfileAndAccumulate();

#ifndef RELEASE
		initProfile.Print();
		kernelProfile.Print();
		pixelProfile.Print();
		pixelHostProfile.Print();
		pixelDeviceProfile.Print();
		ptProfile.Print();
		updateFuncPRofile.Print();
#endif
	}

	__host__ void PathIntegrator::ReserveCancel()
	{
		reserveCancel = true;
	}

	__host__ bool PathIntegrator::IsCancel()
	{
		return !reserveCancel;
	}

	__host__ void PathIntegrator::RenderStraight(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
	}
	__host__ void PathIntegrator::RenderIncremental(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
	}

	__forceinline__ __host__ __device__ void SampleAllInfiniteLight(const IAggregate* dag, const FrameMutableInput& min, const SurfaceIntersection& isect, int threadIndex, OUT ColorRGB& light)
	{
		for (int i = 0; i < min.lightBufferLen; i++)
		{
			switch (min.lightBuffer[i].type)
			{
			case eUnityLightType::Directional:
			{
				if (dag->IntersectOnly(Ray(isect.position, -min.lightBuffer[i].forward), threadIndex))
					continue;

				light = light + fmin(fmax(Dot(isect.normal, -min.lightBuffer[i].forward), 0.f), 1.f) * min.lightBuffer[i].color * min.lightBuffer[i].intensity;
				break;
			}
			case eUnityLightType::Point:
			{
				if (dag->IntersectOnly(Ray(isect.position, (min.lightBuffer[i].position - isect.position).normalized()), threadIndex))
					continue;

				float sqd = (min.lightBuffer[i].position - isect.position).sqrMagnitude();
				if (sqd <= min.lightBuffer[i].range)
					light = light + min.lightBuffer[i].color / sqd * min.lightBuffer[i].intensity;
			}
			break;
			case eUnityLightType::Spot:
			{
				if (dag->IntersectOnly(Ray(isect.position, (min.lightBuffer[i].position - isect.position).normalized()), threadIndex))
					continue;

				Vector3f dir = min.lightBuffer[i].position - isect.position;
				float d = dir.magnitude();
				dir /= d;

				if (d <= min.lightBuffer[i].range &&
					(Dot(dir, min.lightBuffer[i].forward)) < cos(min.lightBuffer[i].angle))
					light = light + min.lightBuffer[i].color / d / d * min.lightBuffer[i].intensity;
			}
			break;
			}
		}
	}

	__forceinline__ __host__ __device__ void SampleSingleLight(
		const IAggregate* dag, const FrameMutableInput& min, const SurfaceIntersection& isect, int threadIndex, Vector3f u, 
		OUT ColorRGB& light, OUT float& pdf
	)
	{
		int i = floorf(u.x * min.lightBufferLen) - 1.f;
		pdf = 1.f;

		switch (min.lightBuffer[i].type)
		{
		case eUnityLightType::Directional:
		{
			float distance = (min.lightBuffer[i].position - isect.position).magnitude();
			if (dag->Intersect(Ray(isect.position, -min.lightBuffer[i].forward), 0, distance, threadIndex, (uint)AggregateItem::Light))
				break;

			light = fmin(fmax(Dot(isect.normal, -min.lightBuffer[i].forward), 0.f), 1.f) * min.lightBuffer[i].color * min.lightBuffer[i].intensity;
			break;
		}
		case eUnityLightType::Point:
		{
			float distance = (min.lightBuffer[i].position - isect.position).magnitude();
			if (dag->Intersect(Ray(isect.position, (min.lightBuffer[i].position - isect.position) / distance), 0, distance, threadIndex, (uint)AggregateItem::Light))
				break;

			//if (distance <= min.lightBuffer[i].range)
				light = min.lightBuffer[i].color /*/ distance*/ * min.lightBuffer[i].intensity;
		}
		break;
		case eUnityLightType::Spot:
		{
			Vector3f dir = min.lightBuffer[i].position - isect.position;
			float distance = dir.magnitude();
			dir = dir / distance;
			if (dag->Intersect(Ray(isect.position, (min.lightBuffer[i].position - isect.position) / distance), 0, distance, threadIndex, (uint)AggregateItem::Light))
				break;

			if (/*d <= min.lightBuffer[i].range &&*/
				(Dot(dir, min.lightBuffer[i].forward)) < cos(min.lightBuffer[i].angle))
				light = min.lightBuffer[i].color /*/ distance / distance*/ * min.lightBuffer[i].intensity;
		}
		break;
		case eUnityLightType::Area:
		{
			Vector3f lightPosition = min.lightBuffer[i].SampleWorldPoint(u.y, u.z);
			float distance = (lightPosition - isect.position).magnitude();

			if (dag->Intersect(Ray(isect.position, (lightPosition - isect.position) / distance), 0, distance, threadIndex, (uint)AggregateItem::Light))
				break;

			light = min.lightBuffer[i].color * min.lightBuffer[i].intensity;
			pdf = min.lightBuffer[i].width * min.lightBuffer[i].height;
		}
		break;
		case eUnityLightType::Disc:
		{
			Vector3f lightPosition = min.lightBuffer[i].SampleWorldPoint(u.y, u.z);
			float distance = (lightPosition - isect.position).magnitude();

			if (dag->Intersect(Ray(isect.position, (lightPosition - isect.position) / distance), 0, distance, threadIndex, (uint)AggregateItem::Light))
				break;

			light = min.lightBuffer[i].color * min.lightBuffer[i].intensity;
			pdf = min.lightBuffer[i].range * min.lightBuffer[i].range;
		}
		break;
		}
	}

	__forceinline__ __host__ __device__ ColorRGB SampleFiniteLight(const FrameMutableInput& min, const Ray & ray, const SurfaceIntersection& isect)
	{
		LightChunk* l = min.lightBuffer + isect.itemIndex;
		float sqd = (l->position - isect.position).sqrMagnitude();
		// TODO:: area light 계산과 다름
		if (sqd <= l->range)
			return l->color / sqd * l->intensity;
		else
			return ColorRGB(0, 0, 0);
	}

	__global__ void InitializeKernel()
	{
		int threadIndex = getGlobalIdx_3D_3D();
		PathThreadSegment& seg = g_PathConstants.segs[threadIndex];

		seg.pixelIndex = threadIndex;

		for (int i = threadIndex; i < g_PathConstants.frameWidth * g_PathConstants.frameHeight; i += g_PathConstants.segCnt)
		{
			g_PathConstants.pixels[i].luminance = ColorRGB::Zero();
			g_PathConstants.pixels[i].throughput = Vector3f::One();
			g_PathConstants.pixels[i].remainSampleCount = g_PathConstants.maxSamplingCount;
			g_PathConstants.pixels[i].remainBounces = g_PathConstants.maxDepthCount;
		}

		curand_init(g_PathConstants.currentTime, threadIndex, 0, g_PathConstants.randStates + threadIndex);

		if (threadIndex == 0)
		{
			*g_PathConstants.devicePixelEndCount = 0;
			*g_PathConstants.nextPixelCount = g_PathConstants.segCnt;
		}
	}

	__global__ void IntersectKernel()
	{
		int iterationCount = 0;
		int threadIndex = getGlobalIdx_3D_3D();
		PathThreadSegment& seg = g_PathConstants.segs[threadIndex];
		if (seg.pixelIndex >= g_PathConstants.frameHeight * g_PathConstants.frameWidth) return;

		while (iterationCount < g_PathConstants.threadIterationCount)
		{
			PathPixelSegment& pixelSeg = g_PathConstants.pixels[seg.pixelIndex];

			pixelSeg.onWorking = 1;
			pixelSeg.finishedWork = 0;

			while (pixelSeg.remainSampleCount > 0)
			{
				pixelSeg.luminance = ColorRGB::Zero();
				pixelSeg.throughput = ColorRGB::One();
				pixelSeg.remainBounces = g_PathConstants.maxDepthCount;
				pixelSeg.flag = 0;

				GetPixelRay(
					g_PathConstants.camera.projectionInverseMatrix,
					g_PathConstants.camera.cameraInverseMatrix,
					seg.pixelIndex,
					Vector2i(g_PathConstants.frameWidth, g_PathConstants.frameHeight),
					pixelSeg.ray
				);

				pixelSeg.flag |= (uint)PathPixelSegmentResult::StartSampling;

				while (pixelSeg.remainBounces > 0)
				{

					pixelSeg.flag |= (uint)PathPixelSegmentResult::StartBounce;

					if (!g_PathConstants.agg->Intersect(pixelSeg.ray, pixelSeg.isect, threadIndex))
					{
						if (pixelSeg.remainBounces == g_PathConstants.maxDepthCount)
						{
							pixelSeg.luminance = pixelSeg.luminance + g_PathConstants.skybox.Sample(pixelSeg.ray, g_PathConstants.immutable.textureBuffer);
						}
						else
						{
							pixelSeg.ray.direction = UniformSampleHemisphereInFrame(
								pixelSeg.isect.normal, 
								Vector2f(
									curand_uniform(g_PathConstants.randStates + threadIndex),
									curand_uniform(g_PathConstants.randStates + threadIndex)
								)
								);

							if (!g_PathConstants.agg->IntersectOnly(pixelSeg.ray, threadIndex))
								pixelSeg.luminance = pixelSeg.luminance + g_PathConstants.skybox.Sample(pixelSeg.ray, g_PathConstants.immutable.textureBuffer) * pixelSeg.throughput;
						}

						pixelSeg.flag |= (uint)PathPixelSegmentResult::TermMiss;
						break;
					}
					else if (pixelSeg.isect.itemIndex < 0)
					{
						pixelSeg.isect.isHit = 0;
						pixelSeg.flag |= (uint)PathPixelSegmentResult::TermError;
						break;
					}

					{
						ColorRGB infLight = ColorRGB::Zero();
						float pdf = 1.f;

						SampleSingleLight(
							g_PathConstants.agg,
							g_PathConstants.mutableIn,
							pixelSeg.isect,
							threadIndex,
							Vector3f(
								curand_uniform(g_PathConstants.randStates + threadIndex),
								curand_uniform(g_PathConstants.randStates + threadIndex),
								curand_uniform(g_PathConstants.randStates + threadIndex)
							),
							infLight,
							pdf
						);

						pixelSeg.luminance = pixelSeg.luminance + infLight / pdf * pixelSeg.throughput;
					}

					Vector3f direction = Vector3f(0, 0, 1.f);
					ColorRGB bxdf = ColorRGB::One();
					float pdf = 0;

					if (pixelSeg.isect.isGeometry)
					{
						MaterialChunk& m = g_PathConstants.mutableIn.materialBuffer[pixelSeg.isect.itemIndex];

						m.GetMaterialInteract(
							g_PathConstants.immutable.textureBuffer,
							pixelSeg.isect,
							pixelSeg.ray,
							g_PathConstants.randStates + threadIndex,
							direction,
							bxdf,
							pdf
						);

						if (pdf == 0)
						{
							pixelSeg.flag |= (uint)PathPixelSegmentResult::TermGeometryPDFZero;
							break;
						}

						pixelSeg.throughput = pixelSeg.throughput * bxdf * Abs(Dot(pixelSeg.ray.direction, pixelSeg.isect.normal)) / pdf;
						if (m.URPLit.IsEmission())
							pixelSeg.luminance = pixelSeg.luminance + pixelSeg.throughput * bxdf;
					}
					else 
					{ 
						LightChunk& l = g_PathConstants.mutableIn.lightBuffer[pixelSeg.isect.itemIndex];
						ColorRGB emittedLight;

						l.GetLightInteract(
							pixelSeg.isect,
							pixelSeg.ray,
							//g_PathConstants.randStates + threadIndex,
							emittedLight,
							pdf
						);

						if (pdf == 0)
						{
							pixelSeg.flag |= (uint)PathPixelSegmentResult::TermLightPDFZero;
							break;
						}
						
						pixelSeg.luminance = pixelSeg.luminance + pixelSeg.throughput * emittedLight / pdf;
						pixelSeg.flag |= (uint)PathPixelSegmentResult::TermHitLight;

						// cannot sampling next ray direction from bxdf
						break;
					}

					pixelSeg.ray.origin = pixelSeg.isect.position;
					pixelSeg.ray.direction = direction;
					pixelSeg.remainBounces--;

					if (pixelSeg.remainBounces == 0)
					{
						pixelSeg.flag |= (uint)PathPixelSegmentResult::TermLimitBounce;
						break;
					}

					if (g_PathConstants.maxDepthCount - pixelSeg.remainBounces > 3 - (g_PathConstants.camera.skyboxIndex >= 0)) {
						float q = fmax(0.05f, 1.f - Luminance(pixelSeg.throughput));
						if (curand_uniform(g_PathConstants.randStates + threadIndex) < q)
						{
							pixelSeg.flag |= (uint)PathPixelSegmentResult::TermRoulette;
							break;
						}
						pixelSeg.throughput = pixelSeg.throughput / (1 - q);
					}
				}

				if (pixelSeg.luminance.r > 0.f || pixelSeg.luminance.g > 0.f || pixelSeg.luminance.b > 0.f)
				{
					int samplingCount = g_PathConstants.cb[seg.pixelIndex].a + 1;
					float prevValueCoeff, currentValueCoeff;

					//if (g_PathConstants.incrementalMode)
					//{
					prevValueCoeff = (float)(samplingCount - 1) / (float)samplingCount;
					currentValueCoeff = 1.f / (float)samplingCount;
					//}
					//else
					//{
					//	prevValueCoeff = 1.f;
					//	currentValueCoeff = 1.f / (float)g_PathConstants.maxSamplingCount;
					//}

					switch (g_PathConstants.pixelContent)
					{
					case PathIntegratorPixelContent::Luminance:
					{
						auto accumulatedValue = pixelSeg.luminance;
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::Bounce:
					{
						auto accumulatedValue = (float)(g_PathConstants.maxDepthCount - pixelSeg.remainBounces) / g_PathConstants.maxDepthCount;
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::ReverseBounce:
					{
						auto accumulatedValue = (float)pixelSeg.remainBounces / g_PathConstants.maxDepthCount;
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::LastThroughput:
					{
						auto accumulatedValue = pixelSeg.throughput;
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::MinThroughput:
					{
						auto accumulatedValue = pixelSeg.throughput;
						g_PathConstants.cb[seg.pixelIndex] = (ColorRGB)Min((Vector3f)g_PathConstants.cb[seg.pixelIndex], (Vector3f)accumulatedValue);
					}
					break;
					case PathIntegratorPixelContent::FlagTermHitLight:
					{
						auto accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermHitLight));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermRoulette:
					{
						float accumulatedValue = (!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermRoulette));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermBlackOut:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermBlackOut));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermLimitBounce:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermLimitBounce));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermMiss:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermMiss));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermError:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermError));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermGeometryPDFZero:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermGeometryPDFZero));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagTermLightPDFZero:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::TermLightPDFZero));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagStartSampling:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::StartSampling));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					case PathIntegratorPixelContent::FlagStartBounce:
					{
						float accumulatedValue = float(!!(pixelSeg.flag & (uint)PathPixelSegmentResult::StartBounce));
						g_PathConstants.cb[seg.pixelIndex] =
							g_PathConstants.cb[seg.pixelIndex] * prevValueCoeff + accumulatedValue * currentValueCoeff;
					}
					break;
					}

					g_PathConstants.cb[seg.pixelIndex].a = samplingCount;
				}

				pixelSeg.remainSampleCount = pixelSeg.remainSampleCount - 1;
				iterationCount++;

				if (iterationCount >= g_PathConstants.threadIterationCount)
					return;
			}

			pixelSeg.onWorking = 0;
			pixelSeg.finishedWork = 1;

			seg.pixelIndex = atomicAdd(g_PathConstants.nextPixelCount, 1);

			if (seg.pixelIndex >= g_PathConstants.frameHeight * g_PathConstants.frameWidth)
				break;
		}
	}

	__global__ void ColorKernel()
	{
		int threadIndex = getGlobalIdx_3D_3D();
		int pixelIndex = threadIndex;
		if (pixelIndex >= g_PathConstants.frameWidth * g_PathConstants.frameHeight) return;

		while (true)
		{
			PathPixelSegment& pixelSeg = g_PathConstants.pixels[pixelIndex];
			ColorRGB infLight = g_PathConstants.skybox.Sample(pixelSeg.ray, g_PathConstants.immutable.textureBuffer);

			infLight = g_PathConstants.skybox.Sample(pixelSeg.ray, g_PathConstants.immutable.textureBuffer);

			if (pixelSeg.isect.isHit && !pixelSeg.isect.isGeometry)
				infLight = infLight + SampleFiniteLight(g_PathConstants.mutableIn, pixelSeg.ray, pixelSeg.isect);

			SampleAllInfiniteLight(
				g_PathConstants.agg,
				g_PathConstants.mutableIn,
				pixelSeg.isect,
				threadIndex,
				infLight
			);

			g_PathConstants.cb[pixelIndex] = g_PathConstants.cb[pixelIndex] * ColorRGBA(infLight);
			g_PathConstants.cb[pixelIndex].a = 1;

			pixelIndex = atomicAdd(g_PathConstants.nextPixelCount, 1);

			if (pixelIndex >= g_PathConstants.frameHeight * g_PathConstants.frameWidth)
				break;
		}
	}

	PathIntegrator* CreatePathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution)
	{
		return new PathIntegrator(param, resultImageResolution);
	}
}
