#include <cstdlib>
#include <chrono>

#include <cuda.h>
#include <curand_kernel.h>

#include "Marshal.h"
#include "Pipeline.h"
#include "DeviceConfig.h"

namespace RadGrabber
{
	struct PathSegment
	{
		Ray ray;
		ColorRGBA attenuation;
		int pixelIndex;
		int remainingBounces;
	};
	/*
		TODO:: BVH Construction in CPU
		TODO:: BVH Traversal in GPU
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
		3. 랜덤 처리 어케함?
			cuRand 사용해야함.


		4. russian roulette 으로 중간에 멈추기 처리?
			구현 이후에

		5. Subsurface Scattering, Transmission 구현은 나중에 ㅎㅎ
	*/

	/*
		Return ray in world space
	*/

	__device__ const double pi = 3.14159265358979323846;

	/*
		Perspective Camera : Generate Ray
	*/
	__forceinline__ __device__ Ray GenerateRay(CameraChunk* c, int pixelIndex, const Vector2i& textureResolution)
	{
		float theta = c->verticalFOV * pi / 180;
		Vector2f size;
		size.y = tan(theta / 2);
		size.x = c->aspect * size.y;
		Vector3f direciton = 
			c->position - size.x * c->right - size.y * c->up - c->forward +
			((float)(pixelIndex % textureResolution.x) / textureResolution.x) * c->right + 
			(float)((int)pixelIndex / textureResolution.x) * c->up - c->position;

		return Ray(c->position, direciton);
	}
	struct BVHNode
	{
		Bounds bound;

		union
		{
			struct
			{
				int splitAxis : 2;
				int nextNode1Index : 31;
				int nextNode2Index : 31;
			};
			int meshRefIndex;
		};
	};


	__forceinline__ __host__ __device__ float gamma(int n) {
		return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
	}

	__noinline__ __device__ bool IntersectMeshLinear(
		IN const Ray& rayInWS, IN MeshRendererChunk* mrc, IN MeshChunk* mc, IN MaterialChunk* matBuffer, 
		OUT SurfaceIntersection* isect, OUT float* distance
	)
	{
		ASSERT_IS_NOT_NULL(mrc);
		ASSERT_IS_NOT_NULL(mc);
		ASSERT_IS_NOT_NULL(matBuffer);
		ASSERT_IS_NOT_NULL(isect);
		ASSERT_IS_NOT_NULL(distance);

		for (int sbidx = 0; sbidx < mc->submeshCount; sbidx++)
		{
			if (mc->submeshArrayPtr[sbidx].bounds.Intersect(rayInWS))
			{
				Ray rayInMS = rayInWS;
				rayInMS.origin -= mrc->position;
				Quaternion q = mrc->quaternion;
				Inverse(q);
				//Rotate(q, rayInMS.origin);
				q = mrc->quaternion;
				Inverse(q);
				//Rotate(q, rayInMS.direction);

				ASSERT((mc->submeshArrayPtr[sbidx].topology == eUnityMeshTopology::Triangles));
				
				int primitiveCount = mc->submeshArrayPtr[sbidx].indexCount / 3, start = mc->submeshArrayPtr[sbidx].indexCount;
				for (int i = 0; i < primitiveCount; i+=3)
				{
					const Vector3f&
						p0 = mc->positions[mc->indices[i + start + 0]],
						p1 = mc->positions[mc->indices[i + start + 1]],
						p2 = mc->positions[mc->indices[i + start + 2]];

					// PBRTv3:: Translate vertices based on ray origin
					Vector3f p0t = p0 - rayInMS.origin;
					Vector3f p1t = p1 - rayInMS.origin;
					Vector3f p2t = p2 - rayInMS.origin;

					// PBRTv3:: Permute components of triangle vertices and ray direction
					int kz = MaxDimension(Abs(rayInMS.direction));
					int kx = kz + 1;
					if (kx == 3) kx = 0;
					int ky = kx + 1;
					if (ky == 3) ky = 0;
					Vector3f d = Permute(rayInMS.direction, kx, ky, kz);
					p0t = Permute(p0t, kx, ky, kz);
					p1t = Permute(p1t, kx, ky, kz);
					p2t = Permute(p2t, kx, ky, kz);

					// PBRTv3:: Apply shear transformation to translated vertex positions
					float Sx = -d.x / d.z;
					float Sy = -d.y / d.z;
					float Sz = 1.f / d.z;
					p0t.x += Sx * p0t.z;
					p0t.y += Sy * p0t.z;
					p1t.x += Sx * p1t.z;
					p1t.y += Sy * p1t.z;
					p2t.x += Sx * p2t.z;
					p2t.y += Sy * p2t.z;

					// PBRTv3:: Compute edge function coefficients _e0_, _e1_, and _e2_
					float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
					float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
					float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

					// PBRTv3:: Fall back to double precision test at triangle edges
					if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
						double p2txp1ty = (double)p2t.x * (double)p1t.y;
						double p2typ1tx = (double)p2t.y * (double)p1t.x;
						e0 = (float)(p2typ1tx - p2txp1ty);
						double p0txp2ty = (double)p0t.x * (double)p2t.y;
						double p0typ2tx = (double)p0t.y * (double)p2t.x;
						e1 = (float)(p0typ2tx - p0txp2ty);
						double p1txp0ty = (double)p1t.x * (double)p0t.y;
						double p1typ0tx = (double)p1t.y * (double)p0t.x;
						e2 = (float)(p1typ0tx - p1txp0ty);
					}

					// PBRTv3:: Perform triangle edge and determinant tests
					if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
						continue;
					float det = e0 + e1 + e2;
					if (det == 0) 
						continue;

					// PBRTv3:: Compute scaled hit distance to triangle and test against ray $t$ range
					p0t.z *= Sz;
					p1t.z *= Sz;
					p2t.z *= Sz;
					float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
					if (det < 0 && (tScaled >= 0/* || tScaled < ray.tMax * det*/))
						continue;
					else if (det > 0 && (tScaled <= 0/* || tScaled > ray.tMax * det*/))
						continue;

					// PBRTv3:: Compute barycentric coordinates and $t$ value for triangle intersection
					float invDet = 1 / det;
					float b0 = e0 * invDet;
					float b1 = e1 * invDet;
					float b2 = e2 * invDet;
					float t = tScaled * invDet;

					// distance check
					if (t > *distance)
						continue;
					*distance = t;

					// PBRTv3:: Ensure that computed triangle $t$ is conservatively greater than zero

					// PBRTv3:: Compute $\delta_z$ term for triangle $t$ error bounds
					float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
					float deltaZ = gamma(3) * maxZt;

					// PBRTv3:: Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
					float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
					float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
					float deltaX = gamma(5) * (maxXt + maxZt);
					float deltaY = gamma(5) * (maxYt + maxZt);

					// PBRTv3:: Compute $\delta_e$ term for triangle $t$ error bounds
					float deltaE =
						2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

					// PBRTv3:: Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
					float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
					float deltaT = 3 *
						(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
						std::abs(invDet);
					if (t <= deltaT) 
						continue;

					//// PBRTv3:: Interpolate $(u,v)$ parametric coordinates and hit point
					//isect->position = b0 * p0 + b1 * p1 + b2 * p2;
					//mrc->quaternion.Rotate(isect->position);
					//isect->position = isect->position + mrc->position;

					//const Vector2f&
					//	uv0 = mc->uvs[mc->indices[i + start + 0]],
					//	uv1 = mc->uvs[mc->indices[i + start + 1]],
					//	uv2 = mc->uvs[mc->indices[i + start + 2]];
					//isect->uv = b0 * uv0 + b1 * uv1 + b2 * uv2;

					//const Vector3f&
					//	n0 = mc->normals[mc->indices[i + start + 0]],
					//	n1 = mc->normals[mc->indices[i + start + 1]],
					//	n2 = mc->normals[mc->indices[i + start + 2]];
					//isect->normal = (b0 * n0 + b1 * n1 + b2 * n2).normalized();
					//mrc->quaternion.Rotate(isect->normal);

					//const Vector3f&
					//	t0 = mc->tangents[mc->indices[i + start + 0]],
					//	t1 = mc->tangents[mc->indices[i + start + 1]],
					//	t2 = mc->tangents[mc->indices[i + start + 2]];
					//isect->tangent = (b0 * n0 + b1 * n1 + b2 * n2).normalized();
					//mrc->quaternion.Rotate(isect->tangent);

					//isect->materialIndex = mrc->materialArrayPtr[sbidx];
				}
			}
		}
			
		return false;
	};

	__noinline__ __device__ bool IntersectGeometryLinear(
		IN const Ray& intersectRay, IN UnityFrameInput* input, OUT SurfaceIntersection& isect
	)
	{
		float distance = FLT_MAX, lastDistance = FLT_MAX;
		SurfaceIntersection isectBuffer;

		for (int i = 0; i < input->meshRendererBufferLen; i++)
		{
			if (input->meshRendererBuffer[i].boundingBox.Intersect(intersectRay))
			{ 
				if (IntersectMeshLinear(intersectRay, input->meshRendererBuffer + i, input->meshBuffer + input->meshRendererBuffer[i].meshRefIndex, input->materialBuffer, &isectBuffer, &lastDistance))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						isect.skinnedRenderer = 0;
						isect.rendererIndex = i;
						distance = lastDistance;
					}
				}
			} 
		}

		for (int i = 0; i < input->skinnedMeshRendererBufferLen; i++)
		{
			if (input->skinnedMeshRendererBuffer[i].boundingBox.Intersect(intersectRay))
			{
				//									   this pointer	      implicate binary compatibilty between MeshRenderer and SkinnedMeshRenderer
				if (IntersectMeshLinear(intersectRay, (MeshRendererChunk*)(input->skinnedMeshRendererBuffer + i), input->meshBuffer + input->skinnedMeshRendererBuffer[i].skinnedMeshRefIndex, input->materialBuffer, &isectBuffer, &lastDistance))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						isect.skinnedRenderer = 1;
						isect.rendererIndex = i;
						distance = lastDistance;
					}
				}
			}
		}

		return distance != FLT_MAX;
	}

	__noinline__ __host__ __device__ ColorRGBA32 GetSkyboxColor(IN const Ray& missedRay, IN SkyboxChunk* skybox)
	{
		/*
			TODO: calculate color for skybox
		*/

		switch (skybox->type)
		{
		case eSkyboxType::Unity6Side:
			break;
		case eSkyboxType::UnityParanomic:
			break;
		case eSkyboxType::UnityProcedural:
			break;
		}

		return ColorRGBA32(0, 0, 0, 0);
	}

	__forceinline__ __device__ void GetAttenuationAndReflectedRay(
		IN const SurfaceIntersection* surf, IN const MaterialChunk* materialBuffer, IN const Texture2DChunk* textureBuffer, 
		IN curandState* randState,  
		INOUT Ray& ray, INOUT Vector4f& attenuation
	)
	{
		switch (materialBuffer[surf->materialIndex].type)
		{
		case eShaderType::UniversalLit:
			materialBuffer[surf->materialIndex].URPLit.GetAttenuation(textureBuffer, surf, randState, attenuation);
			materialBuffer[surf->materialIndex].URPLit.GetRayDirection(textureBuffer, surf, ray.direction);
			break;
		}
	}

	__global__ void IncrementalSamplinByPTInternal(
		IN UnityFrameInput* input, curandState* randomStates, OUT ColorRGBA32* colorBuffer, 
		IN int bufferSize, IN int startPixelIndex, IN int prevSamplingIndex, IN int limitPathLength, 
		IN int selectedCameraIndex, IN Vector2i textureResolution
	)
	{
		int id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y, pixelIndex = startPixelIndex + id,
			pathIndex = 0;
		double pathThroughputWeightByLen = 1.0;

		Ray lastRay = GenerateRay(input->cameraBuffer + selectedCameraIndex, pixelIndex, textureResolution);
		SurfaceIntersection isect;
		PathSegment seg;
		Vector4f attenuation = Vector4f::one();

		while (pathIndex < limitPathLength)
		{
			if (IntersectGeometryLinear(lastRay, input, isect))
			{
				if (isect.emitted)
				{
					// TODO:: isect 이걸로 계산
					break;
				}
				else
				{
					GetAttenuationAndReflectedRay(
						&isect, input->materialBuffer, input->textureBuffer, 
						randomStates + id, lastRay, attenuation
					);
					pathIndex++;
				}
			}
			else
			{
				// TODO:: 스카이박스 + ray.direction(WorldSpace) 으로 계산
				break;
			}
		}

		if (pathIndex == limitPathLength)
		{
			// TODO:: 아무것도 안부딫친 색 계산
		}
	}

	__global__ void InitParams(curandState* states, long long currentTime)
	{
		int pixelIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
		curand_init(currentTime, pixelIndex, 0, states + pixelIndex);
	}

	__host__ int IncrementalPTSampling(IN UnityFrameRequest* hostReq, IN UnityFrameInput* deviceInput)
	{
		int samplingCount = 0, prevDrawSamplingCount = samplingCount;
		const int
			drawSamplingCount = 4,
			limitSamplingCount = hostReq->opt.maxSamplingCount,
			limitPathLength = 50,
			updateMilliSecond = 1000;

		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(0, 4, param);

		curandState* randStateBuffer = nullptr;

		ASSERT_IS_FALSE(cudaMalloc(&randStateBuffer, sizeof(curandState) * param.itemCount));
		ASSERT_IS_FALSE(cudaMemset(randStateBuffer, 0, sizeof(curandState) * param.itemCount));

		ColorRGBA32* deviceColorBuffer = nullptr;

		ASSERT_IS_FALSE(cudaMalloc(&deviceColorBuffer, sizeof(ColorRGBA32) * param.itemCount));
		ASSERT_IS_FALSE(cudaMemset(deviceColorBuffer, 0, sizeof(ColorRGBA32) * param.itemCount));

		std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();

		InitParams<<< param.blockCountInGrid, param.threadCountinBlock, 0, 0 >>>(randStateBuffer, std::chrono::duration_cast<std::chrono::milliseconds>(lastUpdateTime.time_since_epoch()).count());

		while (samplingCount < limitSamplingCount)
		{
			for (int pixelIndex = 0; pixelIndex < hostReq->output.pixelBufferSize.x * hostReq->output.pixelBufferSize.y; pixelIndex += param.itemCount)
			{
				hostReq->output.GetPixelFromTexture(
					deviceColorBuffer,
					pixelIndex,
					param.itemCount,
					[](void* deviceColorBuffer, void* mappedReadBuffer, int pixelIndex, int itemCount, int texItemSize)
					{
						int copySize = texItemSize > pixelIndex + itemCount ? itemCount : texItemSize - (pixelIndex + 1);
						ColorRGBA* colorReadBuffer = (ColorRGBA*)mappedReadBuffer;
						ASSERT_IS_FALSE(cudaMemcpyAsync(deviceColorBuffer, colorReadBuffer + pixelIndex, sizeof(ColorRGBA) * copySize, cudaMemcpyKind::cudaMemcpyHostToDevice, 0));
					}
				);
				
				//IncrementalSamplinByPTInternal <<< param.blockCountInGrid, param.threadCountinBlock, 0, 0 >>> (
				//	deviceInput, randStateBuffer, deviceColorBuffer, param.itemCount, pixelIndex,
				//	samplingCount, limitPathLength, hostReq->opt.selectedCameraIndex, hostReq->output.pixelBufferSize);

				hostReq->output.SetPixelToTexture(
					deviceColorBuffer,
					pixelIndex,
					param.itemCount,
					[](void* deviceColorBuffer, void* mappedWriteBuffer, int pixelIndex, int itemCount, int texItemSize)
					{
						int copySize = texItemSize > pixelIndex + itemCount ? itemCount : texItemSize - (pixelIndex + 1);
						ColorRGBA* colorWriteBuffer = (ColorRGBA*)mappedWriteBuffer;
						ASSERT_IS_FALSE(cudaMemcpyAsync(colorWriteBuffer + pixelIndex, deviceColorBuffer, sizeof(ColorRGBA) * copySize, cudaMemcpyKind::cudaMemcpyDeviceToHost, 0));
					}
				);
			}

			samplingCount++;

			if (samplingCount - prevDrawSamplingCount > drawSamplingCount ||
				std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count() >= updateMilliSecond)
			{
				hostReq->opt.updateFunc();

				prevDrawSamplingCount = drawSamplingCount;
				lastUpdateTime = std::chrono::system_clock::now();
			}
		}

		return 0;
	}
}
