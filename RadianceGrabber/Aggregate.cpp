#include "Aggregate.h"

using namespace RadGrabber;

__host__  LinearAggregate::LinearAggregate(const GeometryInput* param)
{ 
	InitAggregate(param);
}

__host__ LinearAggregate::~LinearAggregate()
{
	if (mMeshArrayAllocated)
	{
		ASSERT_IS_FALSE(cudaFree(mMeshs));
	}
}

__host__ void LinearAggregate::InitAggregate(const GeometryInput* param)
{
	mMeshArrayAllocated = true;

	mMRs = param->meshRendererBuffer;
	mMRCount = param->meshRendererBufferLen;
	mSkinnedMRs = param->skinnedMeshRendererBuffer;
	mSkinnedMRCount = param->skinnedMeshRendererBufferLen;
	mLights = param->lightBuffer;
	mLightCount = param->lightBufferLen;

	mMRMeshCount = param->meshBufferLen;
	mMeshCount = param->meshBufferLen + param->skinnedMeshBufferLen;

	ASSERT_IS_FALSE(cudaMalloc(&mMeshs, mMeshCount * sizeof(MeshChunk)));
	ASSERT_IS_FALSE(
		cudaMemcpy(
			mMeshs, 
			param->meshBuffer, 
			param->meshBufferLen * sizeof(MeshChunk), 
			cudaMemcpyKind::cudaMemcpyHostToHost
		)
	);
	ASSERT_IS_FALSE(
		cudaMemcpy(
			mMeshs + param->meshBufferLen, 
			param->skinnedMeshBuffer, 
			param->skinnedMeshBufferLen * sizeof(MeshChunk), 
			cudaMemcpyKind::cudaMemcpyHostToHost
		)
	);
}	

__device__ bool LinearAggregate::Intersect(const Ray & ray, SurfaceIntersection & isect)
{
	float distance = FLT_MAX, lastDistance = FLT_MAX;
	SurfaceIntersection isectBuffer;

	for (int i = 0; i < mMRCount; i++)
	{
		if (mMRs[i].boundingBox.Intersect(ray))
		{
			if (Intersect(ray, mMRs + i, isectBuffer, lastDistance))
			{
				if (lastDistance < distance)
				{
					isect = isectBuffer;
					isect.isGeometry = 1;
					isect.skinnedRenderer = 0;
					isect.rendererIndex = i;
					distance = lastDistance;
				}
			}
		}
	}

	for (int i = 0; i < mSkinnedMRCount; i++)
	{
		if (mSkinnedMRs[i].boundingBox.Intersect(ray))
		{
			if (Intersect(ray, (MeshRendererChunk*)(mSkinnedMRs + i), isectBuffer, lastDistance))
			{
				if (lastDistance < distance)
				{
					isect = isectBuffer;
					isect.isGeometry = 1;
					isect.skinnedRenderer = 1;
					isect.rendererIndex = i;
					distance = lastDistance;
				}
			}
		}
	}

	for (int i = 0; i < mLightCount; i++)
	{
		if (mLights[i].IntersectRay(ray, isectBuffer, lastDistance))
		{
			if (lastDistance < distance)
			{
				isect = isectBuffer;
				isect.isNotLight = 0;
				distance = lastDistance;
			}
		}
	}

	return distance != FLT_MAX;
}

__forceinline__ __host__ __device__ float gamma(int n) {
	return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

__device__ bool Intersect(const Ray& ray, const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, SurfaceIntersection& isect, float& lastDistance, Vector3f& bc)
{
	// PBRTv3:: Translate vertices based on ray origin
	Vector3f p0t = p0 - ray.origin;
	Vector3f p1t = p1 - ray.origin;
	Vector3f p2t = p2 - ray.origin;

	// PBRTv3:: Permute components of triangle vertices and ray direction
	int kz = MaxDimension(Abs(ray.direction));
	int kx = kz + 1;
	if (kx == 3) kx = 0;
	int ky = kx + 1;
	if (ky == 3) ky = 0;
	Vector3f d = Permute(ray.direction, kx, ky, kz);
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
		return false;
	float det = e0 + e1 + e2;
	if (det == 0)
		return false;

	// PBRTv3:: Compute scaled hit distance to triangle and test against ray $t$ range
	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if (det < 0 && (tScaled >= 0/* || tScaled < ray.tMax * det*/))
		return false;
	else if (det > 0 && (tScaled <= 0/* || tScaled > ray.tMax * det*/))
		return false;

	// PBRTv3:: Compute barycentric coordinates and $t$ value for triangle intersection
	float invDet = 1 / det;
	bc.x = e0 * invDet;
	bc.y = e1 * invDet;
	bc.z = e2 * invDet;
	float t = tScaled * invDet;

	// distance check
	if (t > lastDistance)
		return false;
	lastDistance = t;

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
	
	return t > deltaT;
}

__device__ bool LinearAggregate::Intersect(const Ray & rayInWS, const MeshRendererChunk* mrc, SurfaceIntersection & isect, float& lastDistance)
{
	ASSERT_IS_NOT_NULL(mr);

	MeshChunk* mc = mMeshs + mrc->meshRefIndex;
	bool intersect = false;

	for (int sbidx = 0; sbidx < mc->submeshCount; sbidx++)
	{
		if (mc->submeshArrayPtr[sbidx].bounds.Intersect(rayInWS))
		{
			// Inverse Transform : Translate
			Ray rayInMS = rayInWS;
			rayInMS.origin -= mrc->position;

			// Inverse Transform : Rotation
			Quaternion q = mrc->quaternion;
			Inverse(q);
			Rotate(q, rayInMS.origin);
			q = mrc->quaternion;
			Inverse(q);
			Rotate(q, rayInMS.direction);

			ASSERT((mc->submeshArrayPtr[sbidx].topology == eUnityMeshTopology::Triangles));

			int primitiveCount = mc->submeshArrayPtr[sbidx].indexCount / 3, 
				start = mc->submeshArrayPtr[sbidx].indexStart;
			for (int i = 0; i < primitiveCount; i += 3)
			{
				const Vector3f&
					p0 = mc->positions[mc->indices[i + start + 0]],
					p1 = mc->positions[mc->indices[i + start + 1]],
					p2 = mc->positions[mc->indices[i + start + 2]];
				Vector3f bc;

				if (
						::Intersect
						(
							rayInMS,
							mc->positions[mc->indices[i + start + 0]],
							mc->positions[mc->indices[i + start + 1]],
							mc->positions[mc->indices[i + start + 2]],
							isect,
							lastDistance,
							bc
						)
					)
				{					
					// PBRTv3:: Interpolate $(u,v)$ parametric coordinates and hit point
					isect.position = bc.x * p0 + bc.y * p1 + bc.z * p2;
					Rotate(mrc->quaternion, isect.position);
					isect.position = isect.position + mrc->position;

					const Vector2f&
						uv0 = mc->uvs[mc->indices[i + start + 0]],
						uv1 = mc->uvs[mc->indices[i + start + 1]],
						uv2 = mc->uvs[mc->indices[i + start + 2]];
					isect.uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

					const Vector3f&
						n0 = mc->normals[mc->indices[i + start + 0]],
						n1 = mc->normals[mc->indices[i + start + 1]],
						n2 = mc->normals[mc->indices[i + start + 2]];
					isect.normal = (bc.x * n0 + bc.y * n1 + bc.z * n2).normalized();
					Rotate(mrc->quaternion, isect.normal);

					const Vector3f&
						t0 = mc->tangents[mc->indices[i + start + 0]],
						t1 = mc->tangents[mc->indices[i + start + 1]],
						t2 = mc->tangents[mc->indices[i + start + 2]];
					isect.tangent = (bc.x * t0 + bc.y * t1 + bc.z * t2).normalized();
					Rotate(mrc->quaternion, isect.tangent);

					intersect = true;
				}
			}
		}
	}

	return intersect;
}

__host__ LinearAggregate* LinearAggregate::GetAggregate(const GeometryInput* hostParam, const GeometryInput* deviceParam)
{
	GeometryInput in;
	ASSERT_IS_FALSE(cudaMemcpy(&in, deviceParam, sizeof(GeometryInput), cudaMemcpyKind::cudaMemcpyHostToDevice));

	LinearAggregate aggregateBuffer(&in), *deviceAggreatePtr;
	ASSERT_IS_FALSE(cudaMalloc(&deviceAggreatePtr, sizeof(LinearAggregate)));
	ASSERT_IS_FALSE(cudaMemcpy(deviceAggreatePtr, &aggregateBuffer, sizeof(LinearAggregate), cudaMemcpyKind::cudaMemcpyHostToDevice));

	return deviceAggreatePtr;
}

__host__ TwoLayerBVH::TwoLayerBVH(const GeometryInput* param)
{
	InitAggregate(param);
}

__host__ void TwoLayerBVH::InitAggregate(const GeometryInput* param)
{
	int internalNodeCount = (param->meshRendererBufferLen + param->skinnedMeshRendererBufferLen + param->lightBufferLen - 1);
	ASSERT(internalNodeCount < 0);

	TransformBVHNode *transformNodeArray = nullptr;
	ASSERT_IS_FALSE(cudaMallocHost(&transformNodeArray, sizeof(TransformBVHNode) * internalNodeCount));
	ASSERT_IS_NOT_NULL(transformNodeArray);

	/*
		TODO:: build bvh tree
	*/
}

__device__ bool TwoLayerBVH::Intersect(const Ray & ray, SurfaceIntersection & isect)
{
	/*
		TODO:: bvh traversal
	*/
	return true;
}

