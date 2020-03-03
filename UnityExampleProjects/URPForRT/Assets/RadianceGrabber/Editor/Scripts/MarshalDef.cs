namespace RadGrab
{
    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;
    using System.Runtime.InteropServices;
    using System;
    using UnityEngine.Rendering;

    public interface IMarshal<MarshalType>
    {
        bool MarshalFrom(MarshalType p);
    }
    public interface IMarshal<MarshalType, OutItemType>
    {
        bool MarshalFrom(MarshalType p, List<OutItemType> outList0);
    }
    public interface IMarshal<MarshalType, OutItemType0, OutItemType1>
    {
        bool MarshalFrom(MarshalType p, List<OutItemType0> outList0, List<OutItemType1> outList1);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public struct Bone
    {
        public Vector3 position;
        public Quaternion rotation;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct Texture2DChunk
    {
        public int width;
        public int height;
        public FilterMode filter;
        public int anisotropic;
        public IntPtr pixelPtr;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public struct RangeUInt
    {
        public uint start;
        public uint count;

        public RangeUInt(uint start, uint count)
        {
            this.start = start;
            this.count = count;
        }
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct MeshChunk
    {
        public int vertexCount;
        public int indexCount;
        public int submeshCount;
        public int bindposeCount;
        public IntPtr vertexPositionArrayPtr;
        public IntPtr vertexNormalArrayPtr;
        public IntPtr vertexTangentArrayPtr;
        public IntPtr vertexUVArrayPtr;
        public IntPtr indexArrayPtr;
        public IntPtr submeshArrayPtr;
        public IntPtr bindposeArrayPtr;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct CameraChunk
    {
        public int cullingMask;
        public Vector3 position;
        public Quaternion quaternion;
        public float verticalFOV;
        public float aspect;
        public Vector3 forward;
        public Vector3 right;
        public Vector3 up;
        public Matrix4x4 projectionMatrix;
        public int skyboxIndex;
    }

    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct LightChunk
    {
        [FieldOffset(0)]
        public Vector3 position;
        [FieldOffset(12)]
        public Quaternion quaternion;
        [FieldOffset(28), MarshalAs(UnmanagedType.SysInt)]
        public LightType type;
        [FieldOffset(32)]
        public float intensity;
        [FieldOffset(36)]
        public float indirectMultiplier;
        [FieldOffset(40)]
        public int cullingMask;

        // point, spot, area
        [FieldOffset(44)]
        public float range;
        // spot
        [FieldOffset(48)]
        public float angle;
        // area
        [FieldOffset(48)]
        public float width;
        // area
        [FieldOffset(52)]
        public float height;
    } // size: 56

    public enum ShaderType
    {
        StandardMetallic    = 0x0000,
        StandardSpecular    = 0x0001,
        UniversalLit        = 0x0100,
        UniversalSimpleLit  = 0x0101,
        UniversalBakedLit   = 0x0102,
        UniversalTerrain    = 0x0103,
        Invalidated         = 0xffff,
        //      UniversalUnlit      = xx,
    }

    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public struct StandardMetallicChunk
    {
    }

    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public struct StandardSpecularChunk
    {
    }

    /// <summary>
    /// Universal RP, Lit 
    /// 
    /// -- Options
    ///  flag & 0x01        ? metallic          : specular
    ///  flag & 0x02        ? opaque            : transparent
    /// (flag & 0x0c) >> 2  0 both
    ///                     1 back
    ///                     2 front
    ///  flag & 0x10        ? alpha clipping    : not alpha clipping
    /// 
    /// -- Inputs
    ///  flag & 0x20        ? in source map     : in albedo map
    ///  flag & 0x40        ? emission          : not emission
    ///  
    /// </summary>
    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct URPLitChunk
    {
        [FieldOffset(0)]
        public int flag;
        [FieldOffset(4)]
        public float alphaThreshold;

        [FieldOffset(8)]
        public Vector4 baseMapTint;
        [FieldOffset(24)]
        public float smoothness;
        [FieldOffset(28)]
        public float glossScale;
        [FieldOffset(32)]
        public float bumpScale;
        [FieldOffset(36)]
        public float occlusionScale;
        [FieldOffset(40)]
        public Vector3 emissionTint;

        [FieldOffset(52)]
        public int baseMapIndex;
        // specular or metallic
        [FieldOffset(56)]
        public int smoothMapIndex;
        [FieldOffset(60)]
        public float metallic;
        [FieldOffset(64)]
        public Vector3 specularColor;
        [FieldOffset(76)]
        public int bumpMapIndex;
        [FieldOffset(80)]
        public int occlusionMapIndex;
        [FieldOffset(84)]
        public int emissionMapIndex;

        [FieldOffset(88)]
        public Vector2 scale;
        [FieldOffset(96)]
        public Vector2 offset;
    }

    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct MaterialChunk
    {
        [FieldOffset(0), MarshalAs(UnmanagedType.SysInt)]
        public ShaderType shader;

        [FieldOffset(4)]
        public URPLitChunk univLit;

        [FieldOffset(4)]
        public StandardMetallicChunk stdMetallic;

        [FieldOffset(4)]
        public StandardSpecularChunk stdSpecular;
    } 

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct MeshRendererChunk
    {
        public Vector3 position;
        public Quaternion quaternion;
        public Vector3 scale;
        public int meshRefIndex;

        public Bounds boundingBox;

        public int materialCount;
        public IntPtr materialArrayPtr;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct SkinnedMeshRendererChunk
    {
        public Vector3 position;
        public Quaternion quaternion;
        public Vector3 scale;
        public int skinnedMeshRefIndex;

        public Bounds boundingBox;

        public int materialCount;
        public IntPtr materialArrayPtr;

        public int boneCount;
        public IntPtr boneArrayPtr;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct TerrainRendererChunk
    {
        public Vector3 position;
    }

    public enum SkyboxType
    {
        Unity6Side          = 000,
        UnityCubemap        = 001,
        UnityParanomic      = 002,
        UnityProcedural     = 003,
    }

    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct SkyboxChunk
    {
        [FieldOffset(0)]
        [MarshalAs(UnmanagedType.SysUInt)]
        public SkyboxType type;
        [FieldOffset(4)]
        public Color tintColor;
        [FieldOffset(16)]
        public float exposure;
        [FieldOffset(20)]
        public float rotation;

        // 000. 6Side
        [FieldOffset(24)]
        public int frontTextureIndex;
        [FieldOffset(28)]
        public int backTextureIndex;
        [FieldOffset(32)]
        public int leftTextureIndex;
        [FieldOffset(36)]
        public int rightTextureIndex;
        [FieldOffset(40)]
        public int upTextureIndex;
        [FieldOffset(44)]
        public int downTextureIndex;

        // 001. Cubemap
        //[FieldOffset(24)]
        //public IntPtr frontColors;
        //[FieldOffset(32)]
        //public IntPtr backColors;
        //[FieldOffset(40)]
        //public IntPtr leftColors;
        //[FieldOffset(48)]
        //public IntPtr rightColors;
        //[FieldOffset(56)]
        //public IntPtr upColors;
        //[FieldOffset(64)]
        //public IntPtr downColors;

        // 002. Paranomic
        // flag & 0x01 : 0: Latitude Longitude Layout / 1: 6 Frame Layout
        // flag & 0x02 : 0: 360 Degrees / 1: 180 Degrees
        // flag & 0x04 : mirror on black
        [FieldOffset(24)]
        public int mappingAndImgtypeFlag;
        [FieldOffset(28)]
        public int paranomicIndex;

        // 003. Procedural : ignore rotation
        [FieldOffset(20)]
        public float sunDisk;
        [FieldOffset(24)]
        public float sunSize;
        [FieldOffset(28)]
        public float sunSizeConvergence;
        [FieldOffset(32)]
        public float atmosphereThickness;
        [FieldOffset(36)]
        public Color groundColor;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public unsafe partial struct FrameInput
    {
        public int meshRendererBufferLen;
        public IntPtr meshRendererBuffer;           /*MeshRendererChunk*/
        public int skinnedMeshRendererBufferLen;
        public IntPtr skinnedMeshRendererBuffer;    /*SkinnedMeshREndererChunk*/
        public int lightBufferLen;
        public IntPtr lightBuffer;                  /*LightChunk*/
        public int cameraBufferLen;
        public IntPtr cameraBuffer;                 /*CameraChunk*/
        public int meshBufferLen;
        public IntPtr meshBuffer;                   /*MeshChunk*/
        public int skinnedMeshBufferLen;
        public IntPtr skinnedMeshBuffer;            /*MeshChunk*/
        public int skyboxMaterialBufferLen;
        public IntPtr skyboxMaterialBuffer;         /*SkyboxChunk*/
        public int textureBufferLen;
        public IntPtr textureBuffer;                /*TextureChunk*/
        public int materialBufferLen;
        public IntPtr materialBuffer;               /*MaterialChunk*/
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public unsafe partial struct FrameOutput
    {
        public Vector2Int colorBufferSize;
        public IntPtr colorBuffer;                  /*ID3D11Resource*/
        //public IntPtr updateFuncPtr;                /*UpdateResult*/
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public struct FrameRequestOption
    {
        public Vector2Int resultImageResolution;
        public int selectedCameraIndex;
        public int maxSamplingCount;
        public IntPtr updateFuncPtr;                /*UpdateResult*/
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct UnityFrameRequest
    {
        public FrameRequestOption opt;
        public FrameInput inputData;
        public FrameOutput outputData;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public struct MultiFrameRequestOption
    {
        public Vector2Int resultImageResolution;
        public int selectedCameraIndex;
        public int maxSamplingCount;

        public int totalFrameCount;
        public int framePerSecond;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 4)]
    public partial struct UnityMultiFrameRequest
    {
        public MultiFrameRequestOption opt;
        public FrameInput inputData;
        public FrameOutput outputData;
    }

}
