namespace RadGrab
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Text;
    using Unity.Collections;
    using Unity.Collections.LowLevel.Unsafe;
    using UnityEditor;
    using UnityEngine;
    using UnityEngine.Assertions;
    using UnityEngine.Rendering;

    public unsafe static class TaskController
    {
        [DllImport("KERNEL32.DLL", EntryPoint = "RtlZeroMemory")]
        public static extern bool ZeroMemory(void* destination, int length);
        
        /// <summary>
        /// [In]
        /// </summary>
        public static UnityRuntimeData runtimeData;
        public static UnityFrameRequest req;
        public static UnityRenderingDataBuilder inputBuilder;

        public static Texture2D result;
        public static NativeArray<Color> colorBuffer;

        /// <summary>
        /// [Out]
        /// </summary>
        public static IntPtr outputTexturePtr;

        public static void Initialize()
        {
            if (result != null && colorBuffer != null)
                ZeroMemory(colorBuffer.GetUnsafePtr<Color>(), Marshal.SizeOf<Color>() * result.width * result.height);
        }

        public static void Clear()
        {
        }

        public static int SaveSingleFrame(string fileName, FrameRequestOption opt, UnityRuntimeData runtimeData, int selectedCameraIndex)
        {
            if (!RadGrabberDLL.IsRadianceGrabberLoaded()) return -1;

            req.opt = opt;

            inputBuilder.ConvertRenderingData(runtimeData, ref req.inputData);

            req.inputData.selectedCameraIndex = selectedCameraIndex;

            fixed (UnityFrameRequest* reqPtr = &req)
            {
                int ret = RadGrabberDLL.SaveSingleFrameRequestImpl(fileName, reqPtr);
                if (ret == 0)
                {
                    Debug.LogFormat(
                        "Successive saving single frame data!(ms:{0}, sms:{1}, mr:{2}, smr:{3}, light:{4}, tex:{5}, cam:{6}, mat:{7}, sky:{8})",
                        req.inputData.meshBufferLen, req.inputData.skinnedMeshBufferLen, req.inputData.meshRendererBufferLen,
                        req.inputData.skinnedMeshRendererBufferLen, req.inputData.lightBufferLen, req.inputData.textureBufferLen,
                        req.inputData.cameraBufferLen, req.inputData.materialBufferLen, req.inputData.skyboxMaterialBufferLen
                        );

                    //StringBuilder builder = new StringBuilder("mrs\n");
                    //for (int i = 0; i < runtimeData.mrs.Length; i++)
                    //    builder.Append(i).Append("::").Append(runtimeData.mrs[i].gameObject.name).Append("\n");

                    //Debug.Log(builder.ToString());
                }
                else
                    Debug.LogFormat("Fail, ret code::{0}", ret);

                return ret;
            }
        }

        public static int StartSingleFrameGeneration(FrameRequestOption opt, UnityRuntimeData runtimeData, int selectedCameraIndex)
        {
            req.opt = opt;
            req.inputData.selectedCameraIndex = selectedCameraIndex;
            inputBuilder.ConvertRenderingData(runtimeData, ref req.inputData);

            if (
                (req.outputData.colorBuffer == null || result == null) ||
                opt.resultImageResolution != new Vector2Int(result.width, result.height)
                )
            {
                if (result == null)
                {
                    result = new Texture2D(opt.resultImageResolution.x, opt.resultImageResolution.y, TextureFormat.RGBAFloat, false);
                }
                else
                {
                    result.Resize(opt.resultImageResolution.x, opt.resultImageResolution.y);                   
                }
            }

            colorBuffer = result.GetRawTextureData<Color>();
            ZeroMemory(NativeArrayUnsafeUtility.GetUnsafePtr<Color>(colorBuffer), Marshal.SizeOf<Color>() * opt.resultImageResolution.x * opt.resultImageResolution.y);
            result.Apply();
            req.outputData.colorBuffer = colorBuffer.GetUnsafePtr();

            fixed (UnityFrameRequest* reqPtr = &req)
            {
                return RadGrabberDLL.GenerateSingleFrameIncrementalImpl(reqPtr);
            }
        }

        public static void UploadColorFromNativeArray()
        {
            result.Apply();
        }
        
        public static bool StopSingleFrameGeneration()
        {
            RadGrabberDLL.StopGenerateSingleFrameImpl();
            return true;
        }

        public static bool IsGeneratingSingleFrame()
        {
            return RadGrabberDLL.IsRadianceGrabberLoaded() && RadGrabberDLL.IsSingleFrameGeneratingImpl() != 0;
        }

        public static void StopAllSingleFrameGeneration()
        {
        }

        public static void StartMultiFrameRecord(MultiFrameRequestOption opt, UnityRuntimeData runtimeData)
        {
            TaskController.runtimeData = runtimeData;

        }

        public static void RecordOneFrame()
        {
        }

        public static bool StopMultiFrameRecord(int taskID)
        {
            return true;
        }

        public static int StartMultiFrameGeneration()
        {
            return -1;
        }

        public static bool StopMultiFrameGeneration(int taskID)
        {
            return true;
        }

        [InitializeOnLoadMethod()]
        private static void OnEditorIntilaized()
        {
            inputBuilder = new UnityRenderingDataBuilder();
        }

        [UnityEditor.Callbacks.DidReloadScripts]
        private static void OnScriptsReloaded()
        {
            // do something
        }
    }
}
