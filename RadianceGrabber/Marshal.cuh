
#include "DataTypes.cuh"
#include "Unity/RenderAPI.h"
#include "Util.h"

#pragma once

namespace RadGrabber
{
	/*
		TODO:: 한 프레임 데이터ㅓ와 여러 프레임 데이터의 인터페이스를 추상화 해야함.
	*/

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

#pragma pack(push, 4)

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
		/*
			host : color 2d vector
			device : texture object
		*/
		void*					pixelPtr;
		bool					hasAlpha;
		
		__host__ __device__ Texture2DChunk() {}
		__host__ __device__ Texture2DChunk(Vector2i s, eUnityFilterMode e, int aniso, void* pp) : size(s), filter(e), anisotropic(aniso), pixelPtr(pp) {}
		__host__ __device__ Texture2DChunk(const Texture2DChunk& c) : size(c.size), filter(c.filter), anisotropic(c.anisotropic), pixelPtr(c.pixelPtr) {}

		__host__ __device__ ColorRGBA Sample8888(const Vector2f& uv) const;
		__host__ __device__ ColorRGBA Sample32323232(const Vector2f& uv) const;
	};

	struct MeshChunk
	{
		int									vertexCount;
		int									indexCount;
		int									submeshCount;
		int									bindposeCount;
		Vector3f*							positions;
		Vector3f*							normals;
		Vector4f*							tangents;
		Vector2f*							uvs;
		int*								indices;
		UnitySubMeshDescriptor*				submeshArrayPtr;
		Matrix4x4*							bindposeArrayPtr; // allocated!;
		Bounds								aabbInMS;

		__host__ __device__ MeshChunk() {}
		__host__ __device__ MeshChunk(const MeshChunk& c) : 
			vertexCount(c.vertexCount), indexCount(c.indexCount), submeshCount(c.submeshCount), bindposeCount(c.bindposeCount),
			positions(c.positions), normals(c.normals), tangents(c.tangents), uvs(c.uvs), 
			indices(c.indices), submeshArrayPtr(c.submeshArrayPtr), bindposeArrayPtr(c.bindposeArrayPtr),
			aabbInMS(c.aabbInMS)
		{}

		__host__ __device__ int GetSubmeshIndexFromIndex(int index) const
		{
			for (int i = 0; i < submeshCount; i++)
				if (submeshArrayPtr[i].indexStart <= index && index - submeshArrayPtr[i].indexStart < submeshArrayPtr[i].indexCount)
					return i;

			return -1;
		}
	};

	struct CameraChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		Matrix4x4				transformMatrix;
		Matrix4x4				transformInverseMatrix;
		Vector3f				forward;
		Vector3f				right;
		Vector3f				up;
		Matrix4x4				projectionMatrix;
		Matrix4x4				projectionInverseMatrix;
		Matrix4x4				cameraMatrix;
		Matrix4x4				cameraInverseMatrix;
		float					verticalFOV; // vertical
		float					aspect;
		float					nearClipPlane;
		float					farClipPlane;
		int						skyboxIndex;
		int						layerMask;

		__host__ __device__ CameraChunk() {}
		__host__ __device__ CameraChunk(const CameraChunk& c) : 
			layerMask(c.layerMask), position(c.position), quaternion(c.quaternion), scale(c.scale),
			verticalFOV(c.verticalFOV), aspect(c.aspect), 
			forward(c.forward), right(c.right), up(c.up), skyboxIndex(c.skyboxIndex),
			projectionMatrix(c.projectionMatrix), projectionInverseMatrix(c.projectionInverseMatrix), 
			cameraMatrix(c.cameraMatrix), cameraInverseMatrix(c.cameraInverseMatrix),
			nearClipPlane(c.nearClipPlane), farClipPlane(c.farClipPlane)
		{}

		__host__ __device__ void GetRay(const Vector2f& uv, Ray& r);
		__host__ __device__ void GetPixelRay(int pixelIndex, Vector2i s, Ray& r);
	};

	struct LightChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		Matrix4x4				transformMatrix;
		Matrix4x4				transformInverseMatrix;
		Vector3f				forward;
		eUnityLightType			type;
		Vector3f				color;
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
			type(c.type), position(c.position), quaternion(c.quaternion), scale(c.scale),
			transformMatrix(c.transformMatrix), transformInverseMatrix(c.transformInverseMatrix),
			forward(c.forward), color(c.color),
			intensity(c.intensity), indirectMultiplier(c.indirectMultiplier), cullingMask(c.cullingMask)
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

		__host__ __device__ bool IntersectRay(const Ray & ray, SurfaceIntersection& isect, float& intersectDistance) const;
		__host__ __device__ bool IntersectRayOnly(const Ray & ray) const;

		__host__ __device__ bool GetBoundingBox(Bounds& bb) const;
		__host__ __device__ bool Sample(const Ray& ray, const SurfaceIntersection& isect, Vector3f& color) const;
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
		float					alphaThreshold;

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

		__forceinline__ __device__ __host__ int IsMetallicSetup() const
		{
			return !!(flag & 0x01);
		}

		__forceinline__ __device__ __host__ int IsOpaque() const
		{
			return !(flag & 0x02);
		}

		__forceinline__ __device__ __host__ int IsAlphaClipping() const
		{
			return !(flag & 0x10);
		}

		__forceinline__ __device__ __host__ int IsEmission() const
		{
			return !(flag & 0x40);
		}

		__forceinline__ __host__ __device__ Vector3f SampleAlbedo(IN const Texture2DChunk* textureBuffer, IN const Vector2f& uv) const
		{
			Vector3f a = baseMapTint;
			if (baseMapIndex >= 0)
			{
				ColorRGBA c = textureBuffer[baseMapIndex].Sample8888(uv);
				a.x *= c.r;
				a.y *= c.g;
				a.z *= c.b;
			}
			return a;
		}

		__forceinline__ __host__ __device__ float SampleMetallic(IN const Texture2DChunk* textureBuffer, IN const Vector2f& uv) const
		{
			if (smoothMapIndex >= 0)
				return textureBuffer[smoothMapIndex].Sample8888(uv).r;
			else
				return metallic;
		}

		__forceinline__ __host__ __device__ float SampleOcclusion(IN const Texture2DChunk* textureBuffer, IN const Vector2f& uv) const
		{
			if (smoothMapIndex >= 0)
				return textureBuffer[smoothMapIndex].Sample8888(uv).g;
			else
				return 1.f;
		}
		 
		__forceinline__ __host__ __device__ float SampleSmoothness(IN const Texture2DChunk* textureBuffer, IN const Vector2f& uv) const
		{
			if (flag & 0x20)
			{
				if (baseMapIndex >= 0)
					return smoothness * textureBuffer[baseMapIndex].Sample8888(uv).a;
				else
					return smoothness;
			}
			else
			{
				if (smoothMapIndex >= 0)
					return smoothness * textureBuffer[smoothMapIndex].Sample8888(uv).a;
				else
					return smoothness;
			}
				
		}

		__device__ void GetMaterialInteract(IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state, INOUT Vector4f& attenuPerComp, INOUT Ray& ray) const;
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

		 __device__ void GetMaterialInteract(IN const Texture2DChunk* textureBuffer, IN SurfaceIntersection* isect, IN curandState* state, INOUT Vector4f& attenuPerComp, INOUT Ray& ray) const;
	};

	struct MeshRendererChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		Matrix4x4				transformMatrix;
		Matrix4x4				transformInverseMatrix;
		int						meshRefIndex;
		Bounds					boundingBox;

		int						materialCount;
		int*					materialArrayPtr;

		__host__ __device__ MeshRendererChunk() {}
		__host__ __device__ MeshRendererChunk(const MeshRendererChunk& c) : 
			position(c.position), quaternion(c.quaternion), scale(c.scale), boundingBox(c.boundingBox),
			transformMatrix(c.transformMatrix), transformInverseMatrix(c.transformInverseMatrix),
			meshRefIndex(c.meshRefIndex), materialCount(c.materialCount), materialArrayPtr(c.materialArrayPtr)
			
		{}
	};

	struct SkinnedMeshRendererChunk
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		Matrix4x4				transformMatrix;
		Matrix4x4				transformInverseMatrix;
		int						skinnedMeshRefIndex;
		Bounds					boundingBox;

		int						materialCount;
		void*					materialArrayPtr;

		int						boneCount;
		void*					boneArrayPtr;

		__host__ __device__ SkinnedMeshRendererChunk() {}
		__host__ __device__ SkinnedMeshRendererChunk(const SkinnedMeshRendererChunk& c) :
			position(c.position), quaternion(c.quaternion), scale(c.scale), 
			transformMatrix(c.transformMatrix), transformInverseMatrix(c.transformInverseMatrix),
			boundingBox(c.boundingBox),
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
				int				frontTextureIndex;	// +Z
				int				backTextureIndex;	// -Z
				int				leftTextureIndex;	// +X
				int				rightTextureIndex;	// -X
				int				upTextureIndex;		// +Y
				int				downTextureIndex;	// -Y
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

		__forceinline__ __host__ __device__ bool Sample(const Ray& ray, const Texture2DChunk* c, Vector3f& color)
		{
			Vector3f unitDirection = ray.direction.normalized();
			float t = 0.5 * (unitDirection.y + 1.0);
			color = (1.0f - t) * Vector3f(0.3689999, 0.3489999, 0.3409999) + t * Vector3f(0.839f, 0.949f, 0.992f);
			return true;

			/*
				TODO:: Skybox Sampling
			*/

			switch (type)
			{
			case eSkyboxType::Unity6Side:

				//if (ray.direction.x < ray.direction.y)
				//{
				//	if (ray.direction.y < ray.direction.z) // z
				//	{
				//		;
				//	}
				//	else
				//	{
				//		if (ray.direction.x < ray.direction.y) // y
				//		{
				//		}
				//		else // x
				//		{
				//		}
				//	}
				//}
				//else
				//{
				//	if (ray.direction.x < ray.direction.z) // z
				//	{
				//	}
				//	else
				//	{
				//		if (ray.direction.x < ray.direction.y) // y
				//		{
				//		}
				//		else // x
				//		{
				//		}
				//	}
				//}

				break;
			case eSkyboxType::UnityParanomic:
				if (mappingAndImgtypeFlag & 0x01)
				{
					// TODO:: 6 frame layout
				}
				else
				{
					float theta = acos(ray.direction.y) / -PI;

					if (mappingAndImgtypeFlag & 0x02 && theta > 0.5f)
						return false;

					float phi = atan2(ray.direction.x, -ray.direction.z) / -PI * 0.5f;
					ColorRGBA clr = c[paranomicIndex].Sample8888(Vector2f(phi, theta));
					color = Vector3f(clr.r, clr.g, clr.b);
				}
				break;
			case eSkyboxType::UnityProcedural:
				// TODO:: Procedural Sampling
//				static const float3 kDefaultScatteringWavelength = float3(.65, .57, .475);
//				static const float3 kVariableRangeForScatteringWavelength = float3(.15, .15, .15);
//
//#define OUTER_RADIUS 1.025
//				static const float kOuterRadius = OUTER_RADIUS;
//				static const float kOuterRadius2 = OUTER_RADIUS * OUTER_RADIUS;
//				static const float kInnerRadius = 1.0;
//				static const float kInnerRadius2 = 1.0;
//
//				static const float kCameraHeight = 0.0001;
//
//#define kRAYLEIGH (lerp(0.0, 0.0025, pow(_AtmosphereThickness,2.5)))      // Rayleigh constant
//#define kMIE 0.0010             // Mie constant
//#define kSUN_BRIGHTNESS 20.0    // Sun brightness
//
//#define kMAX_SCATTER 50.0 // Maximum scattering value, to prevent math overflows on Adrenos
//
//				static const float kHDSundiskIntensityFactor = 15.0;
//				static const float kSimpleSundiskIntensityFactor = 27.0;
//
//				static const float kSunScale = 400.0 * kSUN_BRIGHTNESS;
//				static const float kKmESun = kMIE * kSUN_BRIGHTNESS;
//				static const float kKm4PI = kMIE * 4.0 * 3.14159265;
//				static const float kScale = 1.0 / (OUTER_RADIUS - 1.0);
//				static const float kScaleDepth = 0.25;
//				static const float kScaleOverScaleDepth = (1.0 / (OUTER_RADIUS - 1.0)) / 0.25;
//				static const float kSamples = 2.0; // THIS IS UNROLLED MANUALLY, DON'T TOUCH
//
//#define MIE_G (-0.990)
//#define MIE_G2 0.9801
//
//#define SKY_GROUND_THRESHOLD 0.02
//				float getRayleighPhase(float eyeCos2)
//				{
//					return 0.75 + 0.75*eyeCos2;
//				}
//				float getRayleighPhase(Vector3f light, Vector3f ray)
//				{
//					float eyeCos = Dot(light, ray);
//					return getRayleighPhase(eyeCos * eyeCos);
//				}
//
//				float scale(float inCos)
//				{
//					float x = 1.0 - inCos;
//					return 0.25 * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
//				}


				break;
			}

			return true;
		}
	};

	struct FrameMutableInput
	{
		int meshRendererBufferLen;
		MeshRendererChunk* meshRendererBuffer;
		int skinnedMeshRendererBufferLen;
		SkinnedMeshRendererChunk* skinnedMeshRendererBuffer;
		int lightBufferLen;
		LightChunk* lightBuffer;
		int cameraBufferLen;
		CameraChunk* cameraBuffer;
		int skyboxMaterialBufferLen;
		SkyboxChunk* skyboxMaterialBuffer;
		int materialBufferLen;
		MaterialChunk* materialBuffer;
		int selectedCameraIndex;	

		__host__ __device__ FrameMutableInput() {}
	};

	struct FrameImmutableInput
	{
		int meshBufferLen;
		MeshChunk* meshBuffer;
		int skinnedMeshBufferLen;
		MeshChunk* skinnedMeshBuffer;
		int textureBufferLen;
		Texture2DChunk* textureBuffer;

		__host__ __device__ FrameImmutableInput() {}
	};

	class IMultipleInput abstract
	{
	public:
		__host__ __device__ virtual int GetCount() const PURE;
		__host__ __device__ virtual const FrameMutableInput* GetMutable(int i) const PURE;
		__host__ __device__ virtual const FrameImmutableInput* GetImmutable() const PURE;
	};

	struct FrameInputInternal
	{
		union
		{
			struct
			{
				// static per task
				int meshBufferLen;
				MeshChunk* meshBuffer;
				int skinnedMeshBufferLen;
				MeshChunk* skinnedMeshBuffer;
				int textureBufferLen;
				Texture2DChunk* textureBuffer;
				// static per frame 
				int meshRendererBufferLen;
				MeshRendererChunk* meshRendererBuffer;
				int skinnedMeshRendererBufferLen;
				SkinnedMeshRendererChunk* skinnedMeshRendererBuffer;
				int lightBufferLen;
				LightChunk* lightBuffer;
				int cameraBufferLen;
				CameraChunk* cameraBuffer;
				int skyboxMaterialBufferLen;
				SkyboxChunk* skyboxMaterialBuffer;
				int materialBufferLen;
				MaterialChunk* materialBuffer;
				int selectedCameraIndex;
			};
			struct
			{
				FrameImmutableInput immutableInput;
				FrameMutableInput mutableInput;
			};
		};


		__host__ __device__ FrameInputInternal() {}
		__host__ __device__ FrameInputInternal(const FrameInputInternal& c) :
			cameraBufferLen(c.cameraBufferLen),
			skyboxMaterialBufferLen(c.skyboxMaterialBufferLen),
			lightBufferLen(c.lightBufferLen),
			meshBufferLen(c.meshBufferLen),
			skinnedMeshBufferLen(c.skinnedMeshBufferLen),
			meshRendererBufferLen(c.meshRendererBufferLen),
			skinnedMeshRendererBufferLen(c.skinnedMeshRendererBufferLen),
			textureBufferLen(c.textureBufferLen),
			materialBufferLen(c.materialBufferLen),
			cameraBuffer(c.cameraBuffer),
			skyboxMaterialBuffer(c.skyboxMaterialBuffer),
			lightBuffer(c.lightBuffer),
			meshBuffer(c.meshBuffer),
			skinnedMeshBuffer(c.skinnedMeshBuffer),
			meshRendererBuffer(c.meshRendererBuffer),
			skinnedMeshRendererBuffer(c.skinnedMeshRendererBuffer),
			textureBuffer(c.textureBuffer),
			materialBuffer(c.materialBuffer),
			selectedCameraIndex(c.selectedCameraIndex)
		{}
	};

	struct FrameInput : public IMultipleInput
	{
		FrameInputInternal in;

		__host__ __device__ FrameInput() : in() {}
		__host__ __device__ FrameInput(const FrameInput& c) 
		{
			in = c.in;
		}
		__host__ __device__ FrameInput(const FrameInputInternal& c) : in(c)
		{
			in = c;
		}

		__host__ __device__ virtual int GetCount() const 
		{
			return 1; 
		}
		__host__ __device__ const FrameMutableInput* GetMutable(int i) const
		{
			ASSERT_IS_TRUE(i == 0);
			return &in.mutableInput;
		}
		__host__ __device__ const FrameImmutableInput* GetImmutable() const
		{
			return &in.immutableInput;
		}

		__host__ __device__ static FrameMutableInput* GetFrameMutableFromFrame(FrameInput* p)
		{
			return &p->in.mutableInput;
		}
		__host__ __device__ static FrameImmutableInput* GetFrameImmutableFromFrame(FrameInput* p)
		{
			return &p->in.immutableInput;
		}
	};

	struct FrameOutput
	{
		void* pixelBuffer; // ColorRGB*
/*
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
*/
		__host__ __device__ FrameOutput() {}
		__host__ __device__ FrameOutput(const FrameOutput& c) :  pixelBuffer(c.pixelBuffer)
		{}
	};
	struct RequestOption
	{
		Vector2i resultImageResolution;
		int maxSamplingCount;
		int maxDepth;
		void(*updateFunc)(int smampleCount);

		__host__ __device__ RequestOption() {}
		__host__ __device__ RequestOption(const RequestOption& c) :
			resultImageResolution(c.resultImageResolution), maxSamplingCount(c.maxSamplingCount), maxDepth(c.maxDepth), updateFunc(c.updateFunc)
		{}
	};

	class FrameRequestMarshal
	{
	public:
		RequestOption opt;
		FrameInputInternal input;
		FrameOutput output;
	};

	class FrameRequest
	{
	public:
		RequestOption opt;
		FrameInput input;
		FrameOutput output;

		__host__ __device__ FrameRequest() : opt(), input(), output() {}
		__host__ __device__ FrameRequest(const FrameRequest& c) : opt(c.opt), input(c.input), output(c.output) {}
		__host__ __device__ FrameRequest(const FrameRequestMarshal& c) : opt(c.opt), input(c.input), output(c.output) {}
	};

	class MultiFrameInput : IMultipleInput
	{
	public:
		FrameImmutableInput immutable;
		int mutableInputLen;
		FrameMutableInput* mutableInputs;

		__host__ __device__ virtual int GetCount() const
		{
			return mutableInputLen;
		}
		__host__ __device__ const FrameMutableInput* GetMutable(int i) const
		{
			ASSERT(i != 0);
			return mutableInputs + i;
		}
		__host__ __device__ const FrameImmutableInput* GetImmutable() const
		{
			return &immutable;
		}
	};
	struct MultiFrameOutput
	{
	};
	class MultiFrameRequest
	{
	public:
		RequestOption opt;
		MultiFrameInput input;
		MultiFrameOutput output;
	};

	struct PathContant
	{
		int maxDepth;
		Vector2i textureResolution;
	};
	struct CheckItems
	{
		int selectedIndex;
		CameraChunk c;
		PathContant p;
	};
#pragma pack(pop)
}

