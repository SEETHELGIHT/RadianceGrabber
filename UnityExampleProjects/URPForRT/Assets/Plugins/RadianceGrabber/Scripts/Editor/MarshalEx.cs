namespace RadGrab
{
    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;
    using System.Runtime.InteropServices;
    using System;
    using UnityEngine.Rendering;
    using UnityEngine.Assertions;

    internal unsafe static class MarshalUtil
    {
        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(void* dest, void* src, int count);
        [DllImport("kernel32.dll")]
        public static extern void RtlZeroMemory(IntPtr dst, UIntPtr length);

        internal static void SizingArray<Chunk>(ref IntPtr arrayPtr, ref int lastSize, int changedSize)
        {
            if (lastSize > 0 && arrayPtr != IntPtr.Zero)
            {
                arrayPtr = Marshal.ReAllocHGlobal(arrayPtr, (IntPtr)(Marshal.SizeOf<Chunk>() * changedSize));
                RtlZeroMemory(arrayPtr + lastSize * Marshal.SizeOf<Chunk>(), (UIntPtr)(Marshal.SizeOf<Chunk>() * (changedSize - lastSize)));
                lastSize = changedSize;
            }
            else
            {
                arrayPtr = Marshal.AllocHGlobal(Marshal.SizeOf<Chunk>() * changedSize);
                RtlZeroMemory(arrayPtr, (UIntPtr)(Marshal.SizeOf<Chunk>() * changedSize));
                lastSize = changedSize;
            }
        }

        internal unsafe static void ConvertArrayToPtr<NativeChunk, ManagedObj>(ref IntPtr nativeArrayPtr, IList<ManagedObj> managedList, int itemCount, Func<IntPtr, ManagedObj, bool> setAction)
        {
            Assert.IsNotNull(setAction, "ConvertArrayToPtr : Action<IntPtr, ManagedObj, bool> setAction is null..");

            int itemSize = Marshal.SizeOf<NativeChunk>();
            byte* ptr = (byte*)nativeArrayPtr.ToPointer();
            
            for (int i = 0, j = 0; i < itemCount; i++)
            {
                void* pb = (void*)(ptr + itemSize * j);
                if (setAction((IntPtr)pb, managedList[i]))
                    j++;
            }
        }

        internal static int GetIndexAndAppendIfNotExist<ListItem>(this List<ListItem> itemList, ListItem item) where ListItem : UnityEngine.Object
        {
            Assert.IsNotNull(itemList, "GetIndexAndAppendIfNotExist : List<ListItem> itemList is null..");

            if (item == null) return -1;

            int index = itemList.FindIndex(ii => ii.Equals(item));
            if (index < 0)
            {
                index = itemList.Count;
                itemList.Add(item);
            }
                
            return index;
        }
    }

    public unsafe partial struct MeshChunk : IMarshal<Mesh>
    {
        public bool MarshalFrom(Mesh ms)
        {
            ms.RecalculateBounds();
            ms.RecalculateNormals();
            ms.RecalculateTangents();

            vertexCount = ms.vertexCount;

            Array arr = null;
            arr = ms.vertices;
            if (arr.Length > 0)
            {
                MarshalUtil.SizingArray<Vector3>(ref vertexPositionArrayPtr, ref vertexCount, arr.Length);
                fixed (Vector3* vs = (Vector3[])arr)
                    MarshalUtil.CopyMemory(vertexPositionArrayPtr.ToPointer(), vs, Marshal.SizeOf<Vector3>() * vertexCount);
            }
            else
                throw new ArgumentNullException();

            arr = ms.normals;
            if (arr.Length > 0)
            {
                MarshalUtil.SizingArray<Vector3>(ref vertexNormalArrayPtr, ref vertexCount, arr.Length);
                fixed (Vector3* vs = (Vector3[])arr)
                    MarshalUtil.CopyMemory(vertexNormalArrayPtr.ToPointer(), vs, Marshal.SizeOf<Vector3>() * vertexCount);
            }

            arr = ms.tangents;
            if (arr.Length > 0)
            {
                MarshalUtil.SizingArray<Vector4>(ref vertexTangentArrayPtr, ref vertexCount, arr.Length);
                fixed (Vector4* vs = (Vector4[])arr)
                    MarshalUtil.CopyMemory(vertexTangentArrayPtr.ToPointer(), vs, Marshal.SizeOf<Vector4>() * vertexCount);
            }

            arr = ms.uv;
            if (arr.Length > 0)
            {
                MarshalUtil.SizingArray<Vector2>(ref vertexUVArrayPtr, ref vertexCount, arr.Length);
                fixed (Vector2* uv = (Vector2[])arr)
                    MarshalUtil.CopyMemory(vertexUVArrayPtr.ToPointer(), uv, Marshal.SizeOf<Vector2>() * vertexCount);
            }

            arr = ms.triangles;
            indexCount = arr.Length;

            for (int i = 0; i < ms.subMeshCount; i++)
                if (ms.GetTopology(i) != MeshTopology.Triangles)
                    throw new ArgumentException();

            if (arr.Length > 0)
            {
                MarshalUtil.SizingArray<int>(ref indexArrayPtr, ref indexCount, arr.Length);
                fixed (int* idx = (int[])arr)
                    MarshalUtil.CopyMemory(indexArrayPtr.ToPointer(), idx, Marshal.SizeOf<int>() * indexCount);
            }
            else
                throw new ArgumentNullException();

            MarshalUtil.SizingArray<SubMeshDescriptor>(ref submeshArrayPtr, ref submeshCount, ms.subMeshCount);
            SubMeshDescriptor* submeshDescPtr = (SubMeshDescriptor*)submeshArrayPtr;
            for (int i = 0; i < submeshCount; i++)
                submeshDescPtr[i] = ms.GetSubMesh(i);

            Matrix4x4[] bs = ms.bindposes;
            MarshalUtil.SizingArray<Matrix4x4>(ref bindposeArrayPtr, ref bindposeCount, bs.Length);
            fixed (Matrix4x4* bsPtr = bs)
                MarshalUtil.CopyMemory(bsPtr, (void*)bindposeArrayPtr, sizeof(Matrix4x4) * bindposeCount);

            aabbInMS = ms.bounds;

            return true;
        }

        public void FreeGlobalMem()
        {
            Marshal.FreeHGlobal(vertexPositionArrayPtr);
            vertexPositionArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(vertexNormalArrayPtr);
            vertexNormalArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(vertexUVArrayPtr);
            vertexUVArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(indexArrayPtr);
            indexArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(submeshArrayPtr);
            submeshArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(bindposeArrayPtr);
            bindposeArrayPtr = IntPtr.Zero;
        }
    }

    public partial struct CameraChunk : IMarshal<Camera, Material>
    {
        public bool MarshalFrom(Camera camera, List<Material> skyboxList)
        {
            cullingMask = camera.cullingMask;
            position = camera.transform.position;
            quaternion = camera.transform.rotation;
            verticalFOV = camera.fieldOfView;
            aspect = camera.aspect;
            forward = camera.transform.forward;
            right = camera.transform.right;
            up = camera.transform.up;
            projectionMatrix = camera.projectionMatrix;
            projectionInverseMatrix = camera.projectionMatrix.inverse;
            cameraMatrix = camera.worldToCameraMatrix;
            cameraInverseMatrix = camera.cameraToWorldMatrix;
            transformMatrix = camera.transform.localToWorldMatrix;
            transformInverseMatrix = camera.transform.worldToLocalMatrix;
            nearClipPlane = camera.nearClipPlane;
            farClipPlane = camera.farClipPlane;

            Skybox skybox = camera.GetComponent<Skybox>();

            if (skybox != null && skybox.material != null)
                skyboxIndex = skyboxList.FindIndex((sbm) => sbm.Equals(skybox.material));
            else
                skyboxIndex = -1;

            return true;
        }
    }

    public partial struct LightChunk : IMarshal<Light>
    {
        public bool MarshalFrom(Light light)
        {
            position = light.transform.position;
            quaternion = light.transform.rotation;
            scale = light.transform.lossyScale;
            transformMatrix = light.transform.localToWorldMatrix;
            transformInverseMatrix = light.transform.worldToLocalMatrix;
            type = light.type;
            forward = light.transform.forward;
            color = new Vector3(light.color.r, light.color.g, light.color.b);
            intensity = light.intensity;
            indirectMultiplier = light.bounceIntensity;
            cullingMask = light.cullingMask;

            switch (type)
            {
                case LightType.Point:
                    range = light.range;
                    break;
                case LightType.Spot:
                    range = light.range;
                    angle = light.spotAngle;
                    break;
                case LightType.Area:
                    range = light.range;
                    width = light.areaSize.x;
                    height = light.areaSize.y;
                    break;
            }

            return true;
        }
    }

    public unsafe partial struct MeshRendererChunk : IMarshal<MeshRenderer, Mesh, Material>
    {
        public bool MarshalFrom(MeshRenderer mr, List<Mesh> meshList, List<Material> materialList)
        {
            position = mr.transform.position;
            quaternion = mr.transform.rotation;
            scale = mr.transform.lossyScale;

            transformMatrix = mr.localToWorldMatrix;
            transformInverseMatrix = mr.worldToLocalMatrix;

            // mesh add
            MeshFilter mf = mr.GetComponent<MeshFilter>();
            if (mf != null)
                meshRefIndex = meshList.GetIndexAndAppendIfNotExist(mf.sharedMesh);
            if (meshRefIndex < 0)
                return false;

            boundingBox = mr.bounds;

            // material add 
            Material[] mats = mr.sharedMaterials;
            MarshalUtil.SizingArray<int>(ref materialArrayPtr, ref materialCount, mats.Length);
            int* materialArrayIntPtr = (int*)materialArrayPtr;

            for (int idx = 0; idx < mats.Length && idx < mf.sharedMesh.subMeshCount; idx++)
            {
                if (mats[idx] == null)
                    materialArrayIntPtr[idx] = -1;
                else
                    materialArrayIntPtr[idx] = materialList.GetIndexAndAppendIfNotExist(mats[idx]);
            }

            return true;
        }

        public void FreeGlobalMem()
        {
            Marshal.FreeHGlobal(materialArrayPtr);
        }
    }

    public unsafe partial struct SkinnedMeshRendererChunk : IMarshal<SkinnedMeshRenderer, Mesh, Material>
    {
        public bool MarshalFrom(SkinnedMeshRenderer smr, List<Mesh> skinnedMeshList, List<Material> materialList)
        {
            position = smr.transform.position;
            quaternion = smr.transform.rotation;
            scale = smr.transform.lossyScale;

            transformMatrix = smr.transform.localToWorldMatrix;
            transformInverseMatrix = smr.transform.worldToLocalMatrix;

            // mesh add
            skinnedMeshRefIndex = skinnedMeshList.GetIndexAndAppendIfNotExist(smr.sharedMesh);
            if (skinnedMeshRefIndex < 0)
                return false;

            boundingBox = smr.bounds;

            // material add
            Material[] mats = smr.sharedMaterials;
            MarshalUtil.SizingArray<int>(ref materialArrayPtr, ref materialCount, mats.Length);
            int* materialArrayIntPtr = (int*)materialArrayPtr;

            for (int idx = 0; idx < mats.Length && idx < smr.sharedMesh.subMeshCount; idx++)
            {
                if (mats[idx] == null)
                    materialArrayIntPtr[idx] = -1;
                else
                    materialArrayIntPtr[idx] = materialList.GetIndexAndAppendIfNotExist(mats[idx]);
            }

            // bone add
            Transform[] bones = smr.bones;
            MarshalUtil.SizingArray<Bone>(ref boneArrayPtr, ref boneCount, bones.Length);
            MarshalUtil.ConvertArrayToPtr<Bone, Transform>(
                ref boneArrayPtr,
                bones,
                boneCount,
                (vptr, transform) =>
                {
                    Bone* bonePtr = (Bone*)vptr;

                    bonePtr->position = transform.position;
                    bonePtr->rotation = transform.rotation;

                    return true;
                }
                );

            return true;
        }

        public void FreeGlobalMem()
        {
            Marshal.FreeHGlobal(materialArrayPtr);
            materialArrayPtr = IntPtr.Zero;
            Marshal.FreeHGlobal(boneArrayPtr);
            boneArrayPtr = IntPtr.Zero;
        }
    }

    public partial struct MaterialChunk : IMarshal<Material, Texture2D>
    {
        public bool MarshalFrom(Material material, List<Texture2D> textureList)
        {
            Shader shader = material.shader;

            if (shader.name.Equals("Universal Render Pipeline/Lit"))
            {
                this.shader = ShaderType.UniversalLit;
                univLit.MarshalFrom(material, textureList);
                return true;
            }
            else if (shader.name.Equals("Standard"))
            {
                this.shader = ShaderType.StandardMetallic;
                return true;
            }
            else if (shader.name.Equals("Standard (Specular setup)"))
            {
                this.shader = ShaderType.StandardSpecular;
                return true;
            }
            else
                return false;
        }
    }

    public partial struct Texture2DChunk : IMarshal<Texture2D>
    {
        public unsafe bool MarshalFrom(Texture2D texture)
        {
            filter = texture.filterMode;
            anisotropic = texture.anisoLevel;

            Color32[] pixels = null;
            CopyTextureData(texture, ref width, ref height, ref pixels);

            int pixelCount = 0;
            MarshalUtil.SizingArray<Color32>(ref pixelPtr, ref pixelCount, width * height);
            fixed (Color32* pp = pixels)
                MarshalUtil.CopyMemory(pixelPtr.ToPointer(), pp, Marshal.SizeOf<Color32>() * width * height);

            return true;
        }

        public static void CopyTextureData(Texture2D texture, ref int width, ref int height, ref Color32[] pixels)
        {
            RenderTexture temporary = RenderTexture.GetTemporary(texture.width, texture.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            Graphics.Blit(texture, temporary);
            RenderTexture active = RenderTexture.active;
            RenderTexture.active = temporary;
            Texture2D expr_3B = new Texture2D(texture.width, texture.height, TextureFormat.ARGB32, false);
            expr_3B.ReadPixels(new Rect(0f, 0f, (float)temporary.width, (float)temporary.height), 0, 0);
            expr_3B.Apply();
            RenderTexture.active = active;
            RenderTexture.ReleaseTemporary(temporary);
            Color32[] pixels2 = expr_3B.GetPixels32();
            if (pixels2.Length != texture.width * texture.height)
            {
                Debug.LogErrorFormat("Texture buffer size mismatch for {0}: {1}, should be {2} ({3} x {4})", 
                
                    texture.name,
                    pixels2.Length,
                    texture.width * texture.height,
                    texture.width,
                    texture.height
                );
            }
            pixels = new Color32[texture.width * texture.height];
            Array.Copy(pixels2, pixels, Math.Min(texture.width * texture.height, pixels2.Length));
            width = texture.width;
            height = texture.height;
        }

        public void FreeGlobalMem()
        {
            Marshal.FreeHGlobal(pixelPtr);
            pixelPtr = IntPtr.Zero;
        }
    }

    public partial struct URPLitChunk : IMarshal<Material, Texture2D>
    {
        public static readonly int workflowKey = Shader.PropertyToID("_WorkflowMode");
        public static readonly int transparentKey = Shader.PropertyToID("_Surface");
        public static readonly int alphaClipKey = Shader.PropertyToID("_AlphaClip");
        public static readonly int alphaThresholdKey = Shader.PropertyToID("_Cutoff");
        public static readonly int renderFaceKey = Shader.PropertyToID("_Cull");
        public static readonly int baseMapKey = Shader.PropertyToID("_BaseMap");
        public static readonly int baseColorKey = Shader.PropertyToID("_BaseColor");
        public static readonly int smoothnessKey = Shader.PropertyToID("_Smoothness");
        public static readonly int smoothnessScaleKey = Shader.PropertyToID("_GlossMapScale");
        public static readonly int smoothnessChannelKey = Shader.PropertyToID("_SmoothnessTextureChannel");
        public static readonly int metallicKey = Shader.PropertyToID("_Metallic");
        public static readonly int metallicMapKey = Shader.PropertyToID("_MetallicGlossMap");
        public static readonly int specularColorKey = Shader.PropertyToID("_SpecColor");
        public static readonly int specularMapKey = Shader.PropertyToID("_SpecGlossMap");
        public static readonly int bumpScaleKey = Shader.PropertyToID("_BumpScale");
        public static readonly int bumpMapKey = Shader.PropertyToID("_BumpMap");
        public static readonly int occlusionScaleKey = Shader.PropertyToID("_OcclusionStrength");
        public static readonly int occlusionMapKey = Shader.PropertyToID("_OcclusionMap");
        public static readonly int emissionColorKey = Shader.PropertyToID("_EmissionColor");
        public static readonly int emissionMapKey = Shader.PropertyToID("_EmissionMap");

        public bool MarshalFrom(Material material, List<Texture2D> textureList)
        {
            bool
                emissive =
                    material.globalIlluminationFlags > MaterialGlobalIlluminationFlags.None &&
                    material.globalIlluminationFlags < MaterialGlobalIlluminationFlags.EmissiveIsBlack,
                workflow = material.GetInt(workflowKey) > 0;

            flag =
                (workflow ? 1 : 0) << 0 |
                material.GetInt(transparentKey) << 1 |
                material.GetInt(alphaClipKey) << 2 |
                material.GetInt(renderFaceKey) << 3 |
                material.GetInt(smoothnessChannelKey) << 5 |
                (emissive ? 1 : 0) << 6
                ;

            alphaThreshold = material.GetFloat(alphaThresholdKey);
            baseMapTint = material.GetColor(baseColorKey);
            smoothness = material.GetFloat(smoothnessKey);
            glossScale = material.GetFloat(smoothnessScaleKey);
            bumpScale = material.GetFloat(bumpScaleKey);
            occlusionScale = material.GetFloat(occlusionScaleKey);
            emissionTint = (Vector4)material.GetColor(emissionColorKey);

            Texture2D tex = null;

            tex = (Texture2D)material.GetTexture(baseMapKey);
            baseMapIndex = tex != null? textureList.GetIndexAndAppendIfNotExist(tex): -1;

            tex = (Texture2D)material.GetTexture(bumpMapKey);
            bumpMapIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;

            {
                if (workflow)
                    tex = (Texture2D)material.GetTexture(metallicMapKey);
                else
                    tex = (Texture2D)material.GetTexture(specularMapKey);

                if (tex != null)
                {
                    if ((smoothMapIndex = textureList.IndexOf(tex)) < 0)
                    {
                        textureList.Add(tex);
                        smoothMapIndex = textureList.Count;
                    }
                    else
                        smoothMapIndex = -1;
                }
                else
                {
                    smoothMapIndex = -1;

                    if (workflow)
                        metallic = material.GetFloat(metallicKey);
                    else
                        specularColor = (Vector4)material.GetColor(specularColorKey);
                }
            }

            tex = (Texture2D)material.GetTexture(occlusionMapKey);
            occlusionMapIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;

            {
                tex = (Texture2D)material.GetTexture(emissionMapKey);
                if (emissive)
                    emissionMapIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                else
                    emissionMapIndex = -1;
            }

            return true;
        }
    }

    public partial struct SkyboxChunk : IMarshal<Material, Texture2D>
    {
        public static readonly string skyboxShaderNamePrefix = "Skybox/";
        public static readonly string paranomicSkyboxShaderName = "Paranomic";
        public static readonly string cubemapSkyboxShaderName = "Cubemap";
        public static readonly string sixsidedSkyboxShaderName = "6 Sided";
        public static readonly string proceduralSkyboxShaderName = "Procedural";

        public bool MarshalFrom(Material skyboxMaterial, List<Texture2D> textureList)
        {
            string shaderName = skyboxMaterial.shader.name;

            if (!shaderName.Contains(skyboxShaderNamePrefix))
                return false;

            string skyboxShaderName = shaderName.Substring(shaderName.IndexOf("/") + 1);

            if (skyboxShaderName.Contains(proceduralSkyboxShaderName))
            {
                tintColor = skyboxMaterial.GetColor("_SkyTint");
                exposure = skyboxMaterial.GetFloat("_Exposure");
                sunDisk = skyboxMaterial.GetFloat("_SunDisk");
                sunSize = skyboxMaterial.GetFloat("_SunSize");
                sunSizeConvergence = skyboxMaterial.GetFloat("_SunSizeConvergence");
                atmosphereThickness = skyboxMaterial.GetFloat("_AtmosphereThickness");
                groundColor = skyboxMaterial.GetColor("_GroundColor");
            }
            else
            {
                tintColor = skyboxMaterial.GetColor("_Tint");
                exposure = skyboxMaterial.GetFloat("_Exposure");
                rotation = skyboxMaterial.GetFloat("_Rotation");

                Texture2D tex = null;

                if (skyboxShaderName.Contains(sixsidedSkyboxShaderName))
                {
                    type = SkyboxType.Unity6Side;

                    tex = (Texture2D)skyboxMaterial.GetTexture("_FrontTex");
                    frontTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                    tex = (Texture2D)skyboxMaterial.GetTexture("_BackTex");
                    backTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                    tex = (Texture2D)skyboxMaterial.GetTexture("_LeftTex");
                    leftTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                    tex = (Texture2D)skyboxMaterial.GetTexture("_RightTex");
                    rightTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                    tex = (Texture2D)skyboxMaterial.GetTexture("_UpTex");
                    upTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                    tex = (Texture2D)skyboxMaterial.GetTexture("_DownTex");
                    downTextureIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                }
                else if (skyboxShaderName.Contains(cubemapSkyboxShaderName))
                {
                    /*
                        TODO:: skybox:cubemap 
                     */
                    Debug.Assert(false);
                    //type = SkyboxType.UnityCubemap;
                    //Cubemap cub = (Cubemap)skyboxMaterial.GetTexture("_Tex");
                    //CubemapFace

                    //cubemapIndex = textureList.GetIndexAndAppendIfNotExist(tex);
                }
                else if (skyboxShaderName.Contains(paranomicSkyboxShaderName))
                {
                    type = SkyboxType.UnityParanomic;

                    mappingAndImgtypeFlag =
                        (0x01 * skyboxMaterial.GetInt("_Layout")) |
                        (0x02 * skyboxMaterial.GetInt("_Mapping")) |
                        (0x04 * skyboxMaterial.GetInt("_MirrorOnBack"));

                    tex = (Texture2D)skyboxMaterial.GetTexture("_MainTex");
                    paranomicIndex = tex != null ? textureList.GetIndexAndAppendIfNotExist(tex) : -1;
                }
            }

            return true;
        }
    }

    public unsafe class UnityRenderingDataBuilder
    {
        private List<Texture2D> textureList = new List<Texture2D>();
        private List<Material> materialList = new List<Material>();
        private List<Material> skyboxList = new List<Material>();
        private List<Mesh> meshList = new List<Mesh>();
        private List<Mesh> skinnedMeshList = new List<Mesh>();

        public bool ConvertRenderingData(UnityRuntimeData runtimeData, ref FrameInput data)
        {
            MarshalUtil.SizingArray<LightChunk>(ref data.lightBuffer, ref data.lightBufferLen, runtimeData.ls.Length);
            MarshalUtil.ConvertArrayToPtr<LightChunk, Light>(
                ref data.lightBuffer,
                runtimeData.ls,
                data.lightBufferLen,
                (ptr, light) =>
                    ((LightChunk*)ptr.ToPointer())->MarshalFrom(light)
                );

            textureList.Clear();
            materialList.Clear();

            skyboxList.Clear();

            if (RenderSettings.skybox != null)
                skyboxList.Add(RenderSettings.skybox);
            foreach (var sbc in runtimeData.sbs)
                skyboxList.Add(sbc.material);

            MarshalUtil.SizingArray<SkyboxChunk>(ref data.skyboxMaterialBuffer, ref data.skyboxMaterialBufferLen, skyboxList.Count);
            MarshalUtil.ConvertArrayToPtr<SkyboxChunk, Material>(
                ref data.skyboxMaterialBuffer,
                skyboxList,
                data.skyboxMaterialBufferLen,
                (ptr, skyboxMaterial) =>
                    ((SkyboxChunk*)ptr.ToPointer())->MarshalFrom(skyboxMaterial, textureList)
                );

            MarshalUtil.SizingArray<CameraChunk>(ref data.cameraBuffer, ref data.cameraBufferLen, runtimeData.cams.Length);
            MarshalUtil.ConvertArrayToPtr<CameraChunk, Camera>(
                ref data.cameraBuffer,
                runtimeData.cams,
                data.cameraBufferLen,
                (ptr, camera) =>
                    ((CameraChunk*)ptr)->MarshalFrom(camera, skyboxList)
                );

            materialList.Clear();
            skinnedMeshList.Clear();

            MarshalUtil.SizingArray<SkinnedMeshRendererChunk>(ref data.skinnedMeshRendererBuffer, ref data.skinnedMeshRendererBufferLen, runtimeData.smrs.Length);
            MarshalUtil.ConvertArrayToPtr<SkinnedMeshRendererChunk, SkinnedMeshRenderer>(
                ref data.skinnedMeshRendererBuffer,
                runtimeData.smrs,
                data.skinnedMeshRendererBufferLen,
                (ptr, smr) =>
                    ((SkinnedMeshRendererChunk*)ptr)->MarshalFrom(smr, skinnedMeshList, materialList)
                );

            meshList.Clear();

            MarshalUtil.SizingArray<MeshRendererChunk>(ref data.meshRendererBuffer, ref data.meshRendererBufferLen, runtimeData.mrs.Length);
            MarshalUtil.ConvertArrayToPtr<MeshRendererChunk, MeshRenderer>(
                ref data.meshRendererBuffer,
                runtimeData.mrs,
                data.meshRendererBufferLen,
                (ptr, mr) =>
                    ((MeshRendererChunk*)ptr)->MarshalFrom(mr, meshList, materialList)
                );

            // mesh -> all mesh renderer check
            {
                MarshalUtil.SizingArray<MeshChunk>(ref data.meshBuffer, ref data.meshBufferLen, meshList.Count);
                MarshalUtil.ConvertArrayToPtr<MeshChunk, Mesh>(
                    ref data.meshBuffer,
                    meshList,
                    data.meshBufferLen,
                    (ptr, ms) =>
                        ((MeshChunk*)ptr.ToPointer())->MarshalFrom(ms)
                    );
            }

            // skinned mesh -> all skinned mesh renderer check
            {
                MarshalUtil.SizingArray<MeshChunk>(ref data.skinnedMeshBuffer, ref data.skinnedMeshBufferLen, skinnedMeshList.Count);
                MarshalUtil.ConvertArrayToPtr<MeshChunk, Mesh>(
                    ref data.skinnedMeshBuffer,
                    skinnedMeshList,
                    data.skinnedMeshBufferLen,
                    (ptr, ms) =>
                        ((MeshChunk*)ptr.ToPointer())->MarshalFrom(ms)
                    );
            }
            
            // material
            MarshalUtil.SizingArray<MaterialChunk>(ref data.materialBuffer, ref data.materialBufferLen, materialList.Count);
            MarshalUtil.ConvertArrayToPtr<MaterialChunk, Material>(
                ref data.materialBuffer,
                materialList,
                data.materialBufferLen,
                (ptr, material) =>
                    ((MaterialChunk*)ptr)->MarshalFrom(material, textureList)
                );

            // after all material processing
            MarshalUtil.SizingArray<Texture2DChunk>(ref data.textureBuffer, ref data.textureBufferLen, textureList.Count);
            MarshalUtil.ConvertArrayToPtr<Texture2DChunk, Texture2D>(
                ref data.textureBuffer,
                textureList,
                data.textureBufferLen,
                (ptr, texture) =>
                    ((Texture2DChunk*)ptr)->MarshalFrom(texture)
            );

            return true;
        }
    }

    public unsafe partial struct FrameInput
    {
        public void FreeGlobalMem()
        {
            MeshChunk* mc = (MeshChunk*)meshBuffer.ToPointer();
            for (int i = 0; i < meshBufferLen; i++)
                mc[i].FreeGlobalMem();

            MeshChunk* smc = (MeshChunk*)skinnedMeshBuffer.ToPointer();
            for (int i = 0; i < skinnedMeshBufferLen; i++)
                smc[i].FreeGlobalMem();

            Texture2DChunk* t2dc = (Texture2DChunk*)textureBuffer.ToPointer();
            for (int i = 0; i < textureBufferLen; i++)
                t2dc[i].FreeGlobalMem();

            MeshRendererChunk* mr = (MeshRendererChunk*)meshRendererBuffer.ToPointer();
            for (int i = 0; i < meshRendererBufferLen; i++)
                mr[i].FreeGlobalMem();

            SkinnedMeshRendererChunk* smr = (SkinnedMeshRendererChunk*)skinnedMeshRendererBuffer.ToPointer();
            for (int i = 0; i < skinnedMeshRendererBufferLen; i++)
                smr[i].FreeGlobalMem();

            Marshal.FreeHGlobal(cameraBuffer);
            cameraBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(skyboxMaterialBuffer);
            skyboxMaterialBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(lightBuffer);
            lightBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(meshBuffer);
            meshBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(skinnedMeshBuffer);
            skinnedMeshBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(meshRendererBuffer);
            meshRendererBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(skinnedMeshRendererBuffer);
            skinnedMeshRendererBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(textureBuffer);
            textureBuffer = IntPtr.Zero;
            Marshal.FreeHGlobal(materialBuffer);
            materialBuffer = IntPtr.Zero;
        }
    }

    public unsafe partial struct FrameOutput
    {
    }
}
