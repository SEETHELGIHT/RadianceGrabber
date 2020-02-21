#include "DataTypes.cuh"
#include "Unity/RenderAPI.h"

#pragma once

namespace RadGrabber
{
	enum class eUnityVertexAttribute : uint
	{ 
		//
		// 요약:
		//     Vertex position.
		Position = 0,
		//
		// 요약:
		//     Vertex normal.
		Normal = 1,
		//
		// 요약:
		//     Vertex tangent.
		Tangent = 2,
		//
		// 요약:
		//     Vertex color.
		Color = 3,
		//
		// 요약:
		//     Primary texture coordinate (UV).
		TexCoord0 = 4,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord1 = 5,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord2 = 6,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord3 = 7,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord4 = 8,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord5 = 9,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord6 = 10,
		//
		// 요약:
		//     Additional texture coordinate.
		TexCoord7 = 11,
		//
		// 요약:
		//     Bone blend weights for skinned Meshes.
		BlendWeight = 12,
		//
		// 요약:
		//     Bone indices for skinned Meshes.
		BlendIndices = 13
	};
	enum class eUnityVertexAttributeFormat : uint
	{
		//
		// 요약:
		//     32-bit float number.
		Float32 = 0,
		//
		// 요약:
		//     16-bit float number.
		Float16 = 1,
		//
		// 요약:
		//     8-bit unsigned normalized number.
		UNorm8 = 2,
		//
		// 요약:
		//     8-bit signed normalized number.
		SNorm8 = 3,
		//
		// 요약:
		//     16-bit unsigned normalized number.
		UNorm16 = 4,
		//
		// 요약:
		//     16-bit signed normalized number.
		SNorm16 = 5,
		//
		// 요약:
		//     8-bit unsigned integer.
		UInt8 = 6,
		//
		// 요약:
		//     8-bit signed integer.
		SInt8 = 7,
		//
		// 요약:
		//     16-bit unsigned integer.
		UInt16 = 8,
		//
		// 요약:
		//     16-bit signed integer.
		SInt16 = 9,
		//
		// 요약:
		//     32-bit unsigned integer.
		UInt32 = 10,
		//
		// 요약:
		//     32-bit signed integer.
		SInt32 = 11
	};
	enum class eUnityLightType : uint
	{
		Spot = 0,
		Directional = 1,
		Point = 2,
		Area = 3,
		Rectangle = 3,
		Disc = 4
	};
	enum class eUnityMeshTopology : uint
	{ 
		//
		// 요약:
		//     Mesh is made from triangles.
		Triangles = 0,
		//
		// 요약:
		//     Mesh is made from quads.
		Quads = 2,
		//
		// 요약:
		//     Mesh is made from lines.
		Lines = 3,
		//
		// 요약:
		//     Mesh is a line strip.
		LineStrip = 4,
		//
		// 요약:
		//     Mesh is made from points.
		Points = 5
	};

	struct UnityVertexAttributeDescriptor
	{
		eUnityVertexAttribute attribute;
		eUnityVertexAttributeFormat format;
		int dimension;
		int stream;
	};
	struct UnitySubMeshDescriptor
	{
		// get half size
		Bounds bounds;
		eUnityMeshTopology topology;
		int indexStart;
		int indexCount;
		int baseVertex;
		int firstVertex;
		int vertexCount;
	};

	enum class eShaderType		: uint
	{
		Standard_Metallic		= 0x0000,
		Standard_Specular		= 0x0001,
		UniversalLit			= 0x0100,
		UniversalSimpleLit		= 0x0101,
		UniversalBakedLit		= 0x0102,
		UniversalTerrain		= 0x0103,
		//  UniversalUnlit      = xx,
	};
	enum class eSkyboxType		: uint
	{
		Unity6Side				= 0x0000,
		UnityCubemap			= 0x0001,
		UnityParanomic			= 0x0002,
		UnityProcedural			= 0x0003,
	};

	enum class eUnityFilterMode	: uint
	{
		//
		// 요약:
		//     Point filtering - texture pixels become blocky up close.
		Point = 0,
		//
		// 요약:
		//     Bilinear filtering - texture samples are averaged.
		Bilinear = 1,
		//
		// 요약:
		//     Trilinear filtering - texture samples are averaged and also blended between mipmap
		//     levels.
		Trilinear = 2
	};

	struct Bone
	{
		Vector3f				position;
		Quaternion				rotation;

		__host__ __device__ Bone() {}
		__host__ __device__ Bone(Vector3f p, Quaternion r) : position(p), rotation(r) { }
		__host__ __device__ Bone(const Bone& b) : position(b.position), rotation(b.rotation) { }
	};

	struct Texture2DChunk
	{
		Vector2i				size;
		eUnityFilterMode		filter;
		int						anisotropic;
		void*					pixelPtr;
		
		__host__ __device__ Texture2DChunk() {}
		__host__ __device__ Texture2DChunk(Vector2i s, eUnityFilterMode e, int aniso, void* pp) : size(s), filter(e), anisotropic(aniso), pixelPtr(pp) {}
		__host__ __device__ Texture2DChunk(const Texture2DChunk& c) : size(c.size), filter(c.filter), anisotropic(c.anisotropic), pixelPtr(c.pixelPtr) {}
	};

	struct MeshChunk
	{
		int									vertexCount;
		int									indexCount;
		int									submeshCount;
		int									bindposeCount;
		Vector3f*							positions;
		Vector3f*							normals;
		Vector3f*							tangents;
		Vector2f*							uvs;
		int*								indices;
		UnitySubMeshDescriptor*				submeshArrayPtr;
		Matrix4x4*							bindposeArrayPtr; // allocated!

		__host__ __device__ MeshChunk() {}
		__host__ __device__ MeshChunk(const MeshChunk& c) : 
			vertexCount(c.vertexCount), indexCount(c.indexCount), submeshCount(c.submeshCount), bindposeCount(c.bindposeCount),
			positions(c.positions), normals(c.normals), tangents(c.tangents), uvs(c.uvs), 
			indices(c.indices), submeshArrayPtr(c.submeshArrayPtr), bindposeArrayPtr(c.bindposeArrayPtr)
		{}
	};

	struct CameraChunk
	{
		int						layerMask;
		Vector3f				position;
		Quaternion				quaternion;
		float					verticalFOV; // vertical
		float					aspect;
		Vector3f				forward;
		Vector3f				right;
		Vector3f				up;
		Matrix4x4				projectionMatrix;
		int						skyboxIndex;

		__host__ __device__ CameraChunk() {}
		__host__ __device__ CameraChunk(const CameraChunk& c) : 
			layerMask(c.layerMask), position(c.position), quaternion(c.quaternion), 
			verticalFOV(c.verticalFOV), aspect(c.aspect), 
			forward(c.forward), right(c.right), up(c.up),
			projectionMatrix(c.projectionMatrix), skyboxIndex(c.skyboxIndex)
		{}
	};

	struct LightChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		eUnityLightType			type;
		float					intensity;
		float					indirectMultiplier;
		int						cullingMask;

		float					range;
		union
		{
			float				angle;
			float				width;
		};
		float					height;

		__host__ __device__ LightChunk() {}
		__host__ __device__ LightChunk(const LightChunk& c) :
			type(c.type), position(c.position), quaternion(c.quaternion),
			intensity(c.intensity), indirectMultiplier(c.indirectMultiplier), 
			cullingMask(c.cullingMask)
		{
			switch (type)
			{
			case eUnityLightType::Point:
				range = c.range;
				break;
			case eUnityLightType::Spot:
				range = c.range;
				angle = c.angle;
				break;
			case eUnityLightType::Area:
				range = c.range;
				width = c.width;
				height = c.height;
				break;
			}
		}

		__host__ __device__ bool IntersectRay(const Ray & ray, SurfaceIntersection& isect, float& intersectDistance);
		__host__ __device__ bool GetBoundingBox(Bounds& bb);
	};

	/// <summary>
	/// Universal RP, Lit, 
	/// 
	/// -- Options
	///  flag & 0x01        ? metallic          : specular
	///  flag & 0x02        ? opaque            : transparent
	/// Render Face
	/// (flag & 0x0c) >> 2  0 front
	///                     1 back
	///                     2 both
	///  flag & 0x10        ? alpha clipping    : not alpha clipping
	/// 
	/// -- Inputs
	///  flag & 0x20        ? in source map     : in albedo map
	///  flag & 0x40        ? emission          : not emission
	///  
	/// </summary>
	struct URPLitMaterialChunk
	{
		int						flag;

		Vector4f				baseMapTint;
		float					smoothness;
		float					glossScale;
		float					bumpScale;
		float					occlusionScale;
		Vector3f				emissionTint;

		int						baseMapIndex;
		int						smoothMapIndex;
		float					metallic;
		Vector3f				specularColor;

		int						bumpMapIndex;
		int						occlusionMapIndex;
		int						emissionMapIndex;

		Vector2f				scale;
		Vector2f				offset;

		__host__ __device__ URPLitMaterialChunk() {}
		__host__ __device__ URPLitMaterialChunk(const URPLitMaterialChunk& c) :
			flag(c.flag), baseMapTint(c.baseMapTint), smoothness(c.smoothness), glossScale(c.glossScale),
			bumpScale(c.bumpScale), occlusionScale(c.occlusionScale), emissionTint(c.emissionTint),
			baseMapIndex(c.baseMapIndex), smoothMapIndex(c.smoothMapIndex), metallic(c.metallic), specularColor(c.specularColor),
			bumpMapIndex(c.bumpMapIndex), occlusionMapIndex(c.occlusionMapIndex), emissionMapIndex(c.emissionMapIndex),
			scale(c.scale), offset(c.offset)
		{ }

		__forceinline__ __device__ __host__ int IsEmission()
		{
			return !!(flag & 0x40);
		}

		__forceinline__ __device__ void GetAttenuation(
			IN const Texture2DChunk* textureBuffer, IN const SurfaceIntersection* isect,
			IN curandState* state,
			INOUT Vector4f& attenuPerComp) const
		{
			float u01 = curand_uniform(state);

			if (u01 < smoothness)
			{
				/*
					TODO:: specular
				*/
			}
			else
			{
				/*
					TODO:: diffuse
				*/
			}

			/*
				TODO:: Attenuation 계싼
			*/
		}

		__forceinline__ __device__ __host__  void GetRayDirection(IN const Texture2DChunk* textureBuffer, IN const SurfaceIntersection* isect, OUT Vector3f& direction) const
		{
			/*
				TODO:: Ray 방향 계산
			*/
		}
	};

	struct MaterialChunk
	{
		eShaderType type;
		union
		{
			URPLitMaterialChunk URPLit;
		};

		__host__ __device__ MaterialChunk() {}
		__host__ __device__ MaterialChunk(const MaterialChunk& c) 
		{
			switch (type = c.type)
			{
			case eShaderType::UniversalLit:
				URPLit = c.URPLit;
				break;
			}
		}
	};

	struct MeshRendererChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		int						meshRefIndex;
		Bounds					boundingBox;

		int						materialCount;
		int*					materialArrayPtr;

		__host__ __device__ MeshRendererChunk() {}
		__host__ __device__ MeshRendererChunk(const MeshRendererChunk& c) : 
			position(c.position), quaternion(c.quaternion), scale(c.scale), boundingBox(c.boundingBox),
			meshRefIndex(c.meshRefIndex), materialCount(c.materialCount), materialArrayPtr(c.materialArrayPtr) 
		{}
	};

	struct SkinnedMeshRendererChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		int						skinnedMeshRefIndex;
		Bounds					boundingBox;

		int						materialCount;
		void*					materialArrayPtr;

		int						boneCount;
		void*					boneArrayPtr;

		__host__ __device__ SkinnedMeshRendererChunk() {}
		__host__ __device__ SkinnedMeshRendererChunk(const SkinnedMeshRendererChunk& c) :
			position(c.position), quaternion(c.quaternion), scale(c.scale), boundingBox(c.boundingBox),
			skinnedMeshRefIndex(c.skinnedMeshRefIndex), 
			materialCount(c.materialCount), boneCount(c.boneCount),
			materialArrayPtr(c.materialArrayPtr), boneArrayPtr(c.boneArrayPtr)
		{}
	};

	struct SkyboxChunk
	{
		eSkyboxType				type;
		ColorRGBA				tintColor;
		float					exposure;
		float					rotation;

		union
		{
			// 000. 6Side
			struct
			{
				int				frontTextureIndex;
				int				backTextureIndex;
				int				leftTextureIndex;
				int				rightTextureIndex;
				int				upTextureIndex;
				int				downTextureIndex;
			};
			// 001. Cubemap
			//struct
			//{
			//	int				cubemapIndex;
			//};
			// 002. Paranomic
			// flag & 0x01 : 0: Latitude Longitude Layout / 1: 6 Frame Layout
			// flag & 0x02 : 0: 360 Degrees / 1: 180 Degrees
			struct
			{
				int				mappingAndImgtypeFlag;
				int				paranomicIndex;
			};
			// 003. Procedural : ignore rotation
			struct
			{
				float			sunDisk;
				float			sunSize;
				float			sunSizeConvergence;
				float			atmosphereThickness;
				ColorRGBA		groundColor;
			};
		};

		__host__ __device__ SkyboxChunk() {}
		__host__ __device__ SkyboxChunk(const SkyboxChunk& c) : 
			type(c.type), tintColor(c.tintColor), exposure(c.exposure), rotation(c.rotation)
		{
			switch (type)
			{
			case eSkyboxType::Unity6Side:
				frontTextureIndex = c.frontTextureIndex;
				backTextureIndex = c.backTextureIndex;
				leftTextureIndex = c.leftTextureIndex;
				rightTextureIndex = c.rightTextureIndex;
				upTextureIndex = c.upTextureIndex;
				downTextureIndex = c.downTextureIndex;
				break;
			case eSkyboxType::UnityParanomic:
				mappingAndImgtypeFlag = c.mappingAndImgtypeFlag;
				paranomicIndex = c.paranomicIndex;
				break;
			case eSkyboxType::UnityProcedural:
				sunDisk = c.sunDisk;
				sunSize = c.sunSize;
				sunSizeConvergence = c.sunSizeConvergence;
				atmosphereThickness = c.atmosphereThickness;
				groundColor = c.groundColor;
				break;
			}
		}
	};

	struct FrameRequestOption
	{
		Vector2i resultImageResolution;
		int selectedCameraIndex;
		int maxSamplingCount;
		void(*updateFunc)();

		__host__ __device__ FrameRequestOption() {}
		__host__ __device__ FrameRequestOption(const FrameRequestOption& c) :
			resultImageResolution(c.resultImageResolution), selectedCameraIndex(c.selectedCameraIndex), 
			maxSamplingCount(c.maxSamplingCount), updateFunc(c.updateFunc)
		{}
	};

	struct GeometryInput
	{
		int meshBufferLen;
		MeshChunk* meshBuffer;
		int meshRendererBufferLen;
		MeshRendererChunk* meshRendererBuffer;
		int lightBufferLen;
		LightChunk* lightBuffer;
		int cameraBufferLen;
		CameraChunk* cameraBuffer;
		int skinnedMeshBufferLen;
		MeshChunk* skinnedMeshBuffer;
		int skinnedMeshRendererBufferLen;
		SkinnedMeshRendererChunk* skinnedMeshRendererBuffer;

		GeometryInput() {}
	};

	struct FrameInput
	{
		int meshBufferLen;
		MeshChunk* meshBuffer;
		int meshRendererBufferLen;
		MeshRendererChunk* meshRendererBuffer;
		int lightBufferLen;
		LightChunk* lightBuffer;
		int cameraBufferLen;
		CameraChunk* cameraBuffer;
		int skinnedMeshBufferLen;
		MeshChunk* skinnedMeshBuffer;
		int skinnedMeshRendererBufferLen;
		SkinnedMeshRendererChunk* skinnedMeshRendererBuffer;
		int skyboxMaterialBufferLen;
		SkyboxChunk* skyboxMaterialBuffer;
		int textureBufferLen;
		Texture2DChunk* textureBuffer;
		int materialBufferLen;
		MaterialChunk* materialBuffer;

		__host__ __device__ FrameInput() {}
		__host__ __device__ FrameInput(const FrameInput& c) :
			cameraBufferLen(c.cameraBufferLen), skyboxMaterialBufferLen(c.skyboxMaterialBufferLen), lightBufferLen(c.lightBufferLen),
			meshBufferLen(c.meshBufferLen), skinnedMeshBufferLen(c.skinnedMeshBufferLen), meshRendererBufferLen(c.meshRendererBufferLen),
			skinnedMeshRendererBufferLen(c.skinnedMeshRendererBufferLen), textureBufferLen(c.textureBufferLen), materialBufferLen(c.materialBufferLen),
			cameraBuffer(c.cameraBuffer), skyboxMaterialBuffer(c.skyboxMaterialBuffer), lightBuffer(c.lightBuffer),
			meshBuffer(c.meshBuffer), skinnedMeshBuffer(c.skinnedMeshBuffer), meshRendererBuffer(c.meshRendererBuffer),
			skinnedMeshRendererBuffer(c.skinnedMeshRendererBuffer), textureBuffer(c.textureBuffer), materialBuffer(c.materialBuffer)
		{}

		GeometryInput* GetGeometry()
		{
			return reinterpret_cast<GeometryInput*>(this);
		}

		static GeometryInput* GetGeometryFromFrame(FrameInput* p)
		{
			return reinterpret_cast<GeometryInput*>(p);
		}
	};
	struct FrameOutput
	{
		Vector2i pixelBufferSize;
		void* pixelBuffer;

		bool GetPixelFromTexture(void* sourceBuffer, int pixelIndex, int itemCount, DEL void (*getPixelFunc)(void*, void*, int, int, int))
		{
			ASSERT(getPixelFunc);

			getPixelFunc(sourceBuffer, RenderAPI::GetRenderAPI()->BeginReadTexture2D(pixelBuffer), pixelIndex, itemCount, pixelBufferSize.x * pixelBufferSize.y);
			RenderAPI::GetRenderAPI()->EndReadTexture2D(pixelBuffer);
			
			return true;
		}

		bool SetPixelToTexture(void* targetBuffer, int pixelIndex, int itemCount, DEL void(*setPixelFunc)(void*, void*, int, int, int))
		{
			ASSERT(setPixelFunc);

			setPixelFunc(targetBuffer, RenderAPI::GetRenderAPI()->BeginWriteTexture2D(pixelBuffer), pixelIndex, itemCount, pixelBufferSize.x * pixelBufferSize.y);
			RenderAPI::GetRenderAPI()->EndWriteTexture2D(pixelBuffer);

			return true;
		}

		__host__ __device__ FrameOutput() {}
		__host__ __device__ FrameOutput(const FrameOutput& c) :
			pixelBufferSize(c.pixelBufferSize), pixelBuffer(c.pixelBuffer)
		{}
	};
	struct FrameRequest
	{
		FrameRequestOption opt;
		FrameInput input;
		FrameOutput output;

		__host__ __device__ FrameRequest() {}
		__host__ __device__ FrameRequest(const FrameRequest& c) : opt(c.opt), input(c.input), output(c.output)
		{}
	};


	struct TerrainRendererChunk
	{
		Vector3f				position;
	};
}

