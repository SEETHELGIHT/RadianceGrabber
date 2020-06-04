#include <chrono>
#include <ratio>
#include <cuda.h>

#include "Integrator.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{
	/*
		TODO:: Path Tracing 구현
		TODO:: MLT 구현
	*/
	/*
		Path Tracing
		1. 메터리얼 별로 BxDF 설정 이후 ray 방향 정하기
			opaque, transparent 냐에 따라서 달라짐.
			alpha clipping 으로 통과 혹은 처리

			if alpha clipping? && texture sampling == 0: break;

			if emission
				끝, 색 계산 후 마침.

			if opaque
				BRDF 사용하여 ray direction + rgb color
			else
				그냥 통과 + rgb filter(use BTDF 지만)

		2. ray 부딫친 위치에서 빛 샘플링 하기

			광원 + emmision Object 처리
			(현재 광원은 따로 처리 되어 있음)

		4. russian roulette 으로 중간에 멈추기 처리?
			구현 이후에
		5. Subsurface Scattering, Transmission 구현은 나중에 ㅎㅎ
	*/

	__host__ PathIntegrator::PathIntegrator(const RequestOption& opt)
	{
		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(0, param);
		mSegmentCount = param.blockCountInGrid.x * param.threadCountinBlock.x;

		Log("Segment Count::%d\n", mSegmentCount);

		mRandStates = (curandState*)MAllocDevice(sizeof(curandState) * mSegmentCount);
		gpuErrchk(cudaMemset(mRandStates, 0, sizeof(curandState) * mSegmentCount));

		mSegments = (PathSegment2*)MAllocDevice(sizeof(PathSegment2) * mSegmentCount);
		gpuErrchk(cudaMemset(mSegments, 0, sizeof(PathSegment2) * mSegmentCount));

		mHostSegments = (PathSegment2*)malloc(sizeof(PathSegment2) * mSegmentCount);
		memset(mHostSegments, 0, sizeof(PathSegment2) * mSegmentCount); 

		reserveCancel = false;
	}

	__host__ PathIntegrator::~PathIntegrator()
	{
		SAFE_DEVICE_DELETE(mRandStates);
		SAFE_DEVICE_DELETE(mSegments);
		SAFE_HOST_DELETE(mHostSegments);
	}

	__constant__ int g_PathContants[3];
	__host__ void SetPathConstant(PathContant c)
	{
		ASSERT_IS_FALSE(cudaMemcpyToSymbol(g_PathContants, &c, sizeof(PathContant)));
	}
	__host__ void SetPathConstant(PathContant* pc)
	{
		ASSERT_IS_FALSE(cudaMemcpyToSymbol(g_PathContants, pc, sizeof(PathContant)));
	}

	__host__ void PathIntegrator::Render(const IAggregate& scene, const IMultipleInput& hin, const IMultipleInput& din, const RequestOption& opt, IColorTarget& target, OptimalLaunchParam& param)
	{
		{
			PathContant initParam;
			mMaxDepth = initParam.maxDepth = opt.maxDepth;
			initParam.textureResolution = opt.resultImageResolution;

			gpuErrchk(cudaMemcpyToSymbol(g_PathContants, &initParam, sizeof(PathContant)));
		}

		int samplingCount = 0, 
			prevDrawSamplingCount = samplingCount,
			framePixelCount = opt.resultImageResolution.x * opt.resultImageResolution.y;
		const int
			drawSamplingCount = 4,
			updateMilliSecond = 500;

		int maxSegmentPixelRefIndex = mSegmentCount, endOfPixelSegmentCount = 0;
		std::chrono::system_clock::time_point 
			lastUpdateTime = std::chrono::system_clock::now(),
			bufferSampleTime = std::chrono::system_clock::now();
		long long 
			kernelLaunchTime = 0,
			cpuTime = 0;

		for (int i = 0; i < hin.GetCount(); i++)
		{
			Log("Mutable Index::%d, Start\n", i);

			const FrameMutableInput* min = hin.GetMutable(i);

			while (samplingCount < opt.maxSamplingCount)
			{
				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}
					

				Log("Sampling::%d, Start\n", samplingCount);

				if (reserveCancel)
				{
					reserveCancel = false;
					return;
				}

				Initialize << <param.blockCountInGrid, param.threadCountinBlock >> >
					(
						0,
						framePixelCount,
						mSegments,
						mRandStates,
						std::chrono::duration_cast<std::chrono::milliseconds>(lastUpdateTime.time_since_epoch()).count(),
						din,
						i
						);

#ifndef RELEASE
				gpuErrchk(cudaMemcpy(mHostSegments, mSegments, mSegmentCount * sizeof(PathSegment2), cudaMemcpyKind::cudaMemcpyDeviceToHost));
#endif

				maxSegmentPixelRefIndex = mSegmentCount;
				endOfPixelSegmentCount = 0;

				while (true)
				{					
					bufferSampleTime = std::chrono::system_clock::now();

					if (reserveCancel)
					{
						reserveCancel = false;
						return;
					}
					IntersectTest<<<param.blockCountInGrid, param.threadCountinBlock, 0, 0>>>
						(
							0,
							mSegmentCount,
							mSegments,
							scene
						);

					gpuErrchk(cudaGetLastError());
#ifndef RELEASE
					if (reserveCancel)
					{
						reserveCancel = false;
						return;
					}
					gpuErrchk(cudaMemcpy(mHostSegments, mSegments, mSegmentCount * sizeof(PathSegment2), cudaMemcpyKind::cudaMemcpyDeviceToHost));
#endif
					
					ScatteringAndAccumAttenuation<<<param.blockCountInGrid, param.threadCountinBlock, 0, 0>>>
						(
							scene,
							0,
							mSegmentCount,
							mSegments,
							mRandStates,
							samplingCount + 1,
							din,
							i,
							target
						);

					gpuErrchk(cudaGetLastError());

					// CPU segments
					{
						if (reserveCancel)
						{
							reserveCancel = false;
							return;
						}
						gpuErrchk(cudaMemcpy(mHostSegments, mSegments, mSegmentCount * sizeof(PathSegment2), cudaMemcpyKind::cudaMemcpyDeviceToHost));

						kernelLaunchTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - bufferSampleTime).count();
						bufferSampleTime = std::chrono::system_clock::now();

						for (int i = 0; i < mSegmentCount; i++)
						{
							if (mHostSegments[i].pixelIndex >= 0)
							{
								if (mHostSegments[i].endOfRay)
								{
									target.UpdateColorFromHost(mHostSegments[i].pixelIndex, samplingCount + 1, (ColorRGBA)mHostSegments[i].attenuation);
									endOfPixelSegmentCount++;

									if (maxSegmentPixelRefIndex < framePixelCount)
									{
										mHostSegments[i].pixelIndex = maxSegmentPixelRefIndex++;
										mHostSegments[i].endOfRay = 0;
										mHostSegments[i].remainingBounces = mMaxDepth;
										mHostSegments[i].attenuation = Vector4f::One();

										min->cameraBuffer[min->selectedCameraIndex].GetPixelRay(mHostSegments[i].pixelIndex, opt.resultImageResolution, mHostSegments[i].ray);

										mHostSegments[i].position =
											mHostSegments[i].normal = 
											mHostSegments[i].tangent = 
											mHostSegments[i].color = Vector3f::Zero();
										mHostSegments[i].isNotLight = mHostSegments[i].lightIndex = 0;
									}
									else
									{
										mHostSegments[i].pixelIndex = -1;
									}
								}
								else
									;
							}
						}

						Log("%d,%d/%d", endOfPixelSegmentCount, maxSegmentPixelRefIndex, framePixelCount);

						if (reserveCancel)
						{
							reserveCancel = false;
							return;
						}
						gpuErrchk(cudaMemcpy(mSegments, mHostSegments, mSegmentCount * sizeof(PathSegment2), cudaMemcpyKind::cudaMemcpyHostToDevice));

						if (endOfPixelSegmentCount >= framePixelCount)
							break;
					}

					cpuTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - bufferSampleTime).count();
					bufferSampleTime = std::chrono::system_clock::now();

					Log("kerneltime::%d(%.2f), cputime::%d(%.2f)\n", kernelLaunchTime, (float)kernelLaunchTime / (kernelLaunchTime + cpuTime), cpuTime, (float)cpuTime / (kernelLaunchTime + cpuTime));
				}

				samplingCount++;

				if (samplingCount - prevDrawSamplingCount > drawSamplingCount ||
					std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count() >= updateMilliSecond)
				{	
					if (opt.updateFunc) opt.updateFunc(samplingCount);

					prevDrawSamplingCount = drawSamplingCount;
					lastUpdateTime = std::chrono::system_clock::now();

					Log("Sampling Refresh::%d\n", samplingCount);
				}

				Log("Sampling::%d, End\n", samplingCount);
			}

			Log("Mutable Index::%d, End\n", i);
		}
	}

	__host__ void PathIntegrator::ReserveCancel()
	{
		reserveCancel = true;
	}

	__host__ bool PathIntegrator::IsCancel()
	{
		return !reserveCancel;
	}

	__host__ __device__ void SampleSkyboxAndInfLight(const IAggregate& dag, const FrameMutableInput* min, const FrameImmutableInput* imin, const Ray & ray, const SurfaceIntersection& isect, int threadIndex, OUT Vector3f& sat, OUT Vector3f& lat)
	{
		SkyboxChunk* s = min->skyboxMaterialBuffer + min->cameraBuffer[min->selectedCameraIndex].skyboxIndex;
		s->Sample(ray, imin->textureBuffer, sat);

		if (isect.isGeometry)
		{
			for (int i = 0; i < min->lightBufferLen; i++)
			{
				switch (min->lightBuffer[i].type)
				{
				case eUnityLightType::Directional:
				{
					if (dag.IntersectOnly(Ray(isect.position, -min->lightBuffer[i].forward), threadIndex))
						continue;

					lat = lat + fmin(fmax(Dot(isect.normal, -min->lightBuffer[i].forward), 0.f), 1.f) * min->lightBuffer[i].color * min->lightBuffer[i].intensity;
					break;
				}
				case eUnityLightType::Point:
				{
					if (dag.IntersectOnly(Ray(isect.position, (min->lightBuffer[i].position - isect.position).normalized()), threadIndex))
						continue;

					float sqd = (min->lightBuffer[i].position - isect.position).sqrMagnitude();
					if (sqd <= min->lightBuffer[i].range)
						lat = lat + min->lightBuffer[i].color / sqd * min->lightBuffer[i].intensity;
				}
				break;
				case eUnityLightType::Spot:
				{
					if (dag.IntersectOnly(Ray(isect.position, (min->lightBuffer[i].position - isect.position).normalized()), threadIndex))
						continue;

					Vector3f dir = min->lightBuffer[i].position - isect.position;
					float d = dir.magnitude();
					dir /= d;

					if (d <= min->lightBuffer[i].range &&
						(Dot(dir, min->lightBuffer[i].forward)) < cos(min->lightBuffer[i].angle))
						lat = lat + min->lightBuffer[i].color / d / d * min->lightBuffer[i].intensity;
				}
				break;
				}
			}
		}
	}

	__host__ __device__ Vector3f SampleFiniteLight(const FrameMutableInput* min, const Ray & ray, const SurfaceIntersection& isect, int threadIndex)
	{
		LightChunk* l = min->lightBuffer + isect.lightIndex;
		float sqd = (l->position - isect.position).sqrMagnitude();
		// TODO:: area light 계산과 다름
		if (sqd <= l->range)
			return l->color / sqd * l->intensity;
		else
			return Vector3f::Zero();
	}

	__global__ void Initialize(int threadOffset, int pixelItemCount, PathSegment2* segments, /*Ray* rays, SurfaceIntersection* isects, */curandState* states, long long currentTime, const IMultipleInput& din, int mutableIndex)
	{
		int threadIndex = threadIdx.x + blockIdx.x * blockDim.x + threadOffset;

		if (threadIndex >= pixelItemCount)
		{
			segments[threadIndex].pixelIndex = -1;
			return;
		}
		
		curand_init(currentTime, threadIndex, 0, states + threadIndex);

		segments[threadIndex].pixelIndex = threadIndex;
		segments[threadIndex].endOfRay = 0;
		segments[threadIndex].attenuation = Vector4f::One();
		segments[threadIndex].remainingBounces = g_PathContants[0];

		// Generate Ray
		const FrameMutableInput* in = din.GetMutable(mutableIndex); 
		CameraChunk& c = in->cameraBuffer[in->selectedCameraIndex];
		c.GetPixelRay(segments[threadIndex].pixelIndex, Vector2i(g_PathContants[1], g_PathContants[2]), segments[threadIndex].ray);

		segments[threadIndex].position = segments[threadIndex].normal = segments[threadIndex].tangent = segments[threadIndex].color = Vector3f::Zero();
		segments[threadIndex].isNotLight = segments[threadIndex].lightIndex = 0;
	}

	__global__ void IntersectTest(int threadOffset, int segmentCount, PathSegment2* segments, /*Ray* rays, SurfaceIntersection* isects, */const IAggregate& scene)
	{
		int threadIndex = threadIdx.x + blockIdx.x * blockDim.x + threadOffset;

		if (segments[threadIndex].pixelIndex < 0) return;
		if (segments[threadIndex].foundLightSourceFlag ||
			segments[threadIndex].endOfDepthFlag) return;

		if (scene.Intersect(segments[threadIndex].ray, segments[threadIndex].isect, threadIndex))
			segments[threadIndex].remainingBounces--;
		else
			segments[threadIndex].remainingBounces = -1;
	}

 	__global__ void ScatteringAndAccumAttenuation(const IAggregate& dag, int threadOffset, int segmentCount, PathSegment2* segments, curandState* states, int samplingCount, const IMultipleInput& din, int mutableIndex, IColorTarget& target)
	{
		int threadIndex = threadIdx.x + blockIdx.x * blockDim.x + threadOffset;
		
		if (segments[threadIndex].pixelIndex < 0) return;

		if (segments[threadIndex].remainingBounces >= 0)
		{
			if (segments[threadIndex].isGeometry)
			{		
				const FrameMutableInput* min = din.GetMutable(mutableIndex);
				const FrameImmutableInput* imin = din.GetImmutable();
				MaterialChunk& m = min->materialBuffer[segments[threadIndex].materialIndex];
				
				m.GetMaterialInteract(
					imin->textureBuffer, 
					&segments[threadIndex].isect,
					states + threadIndex, 
					segments[threadIndex].attenuation, 
					segments[threadIndex].ray
				);
			}
			else
			{
				segments[threadIndex].foundLightSourceFlag = 1;

				Vector3f lightColor = SampleFiniteLight(
					din.GetMutable(mutableIndex), 
					segments[threadIndex].ray,
					segments[threadIndex].isect,
					threadIndex
				);
				segments[threadIndex].attenuation = segments[threadIndex].attenuation.GetVec3() * lightColor;
			}

			if (segments[threadIndex].remainingBounces == 0)
			{
				segments[threadIndex].endOfDepthFlag = 1;
				goto END_COLOR_SAMPLING_WITHOUT_FLAG;
			}
		}
		else if (segments[threadIndex].remainingBounces < 0)
		{
			segments[threadIndex].missedRay = 1;
			
		END_COLOR_SAMPLING_WITHOUT_FLAG:
			Vector3f skyboxColor, lightColor;
			SampleSkyboxAndInfLight(dag, din.GetMutable(mutableIndex), din.GetImmutable(), segments[threadIndex].ray, segments[threadIndex].isect, threadIndex, skyboxColor, lightColor);

			segments[threadIndex].attenuation = segments[threadIndex].attenuation.GetVec3() * (/*lightColor + */skyboxColor);
		}
		
	}

	PathIntegrator* CreatePathIntegrator(const RequestOption& opt)
	{
		return new PathIntegrator(opt);
	}

	__host__ void MLTIntergrator::Render(const IAggregate& scene, const IMultipleInput& hin, const IMultipleInput& din, const RequestOption& opt, IColorTarget& target, OptimalLaunchParam& param)
	{
	}


	MLTIntergrator* CreateMLTIntegrator(const RequestOption& opt) { return nullptr; }
}
