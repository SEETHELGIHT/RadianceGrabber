namespace RadGrab
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using UnityEditor;
    using UnityEngine;
    using UnityEngine.Assertions;
    using UnityEngine.Rendering;

    public unsafe static class TaskController
    {
        //[DllImport("RadianceGrabber")]
        //private static extern void StartCalc(Config* cfg);
        //[DllImport("RadianceGrabber")]
        //private static extern void StopCalc();
        [DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private static extern int GenerateSingleFrame(UnityFrameRequest* req);
        [DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private static extern int StopAll();
        //[DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        //private static extern int StopGenerateSingleFrame(int taskID);
        //[DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        //private static extern int StopRecordMultiFrame(int taskID);
        //[DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        //private static extern int StopCalculateMultiFrame(int taskID);

        /// <summary>
        /// [In]
        /// </summary>
        public static UnityRuntimeData runtimeData;
        public static UnityFrameRequest req;
        public static UnityRenderingDataBuilder inputBuilder;

        public static Texture2D result;

        /// <summary>
        /// [Out]
        /// </summary>
        public static IntPtr outputTexturePtr;

        public static int StartSingleFrameGeneration(FrameRequestOption opt, UnityRuntimeData runtimeData)
        {
            req.opt = opt;

            inputBuilder.ConvertRenderingData(runtimeData, ref req.inputData);

            if (
                (req.outputData.colorBuffer == null || result == null) ||
                opt.resultImageResolution != new Vector2Int(result.width, result.height)
                )
            {
                if (result == null)
                    result = new Texture2D(1920, 1080, TextureFormat.RGBA32, false);
                else
                    result.Resize(opt.resultImageResolution.x, opt.resultImageResolution.y);

                req.outputData.Init(result);
            }

            fixed (UnityFrameRequest* reqPtr = &req)
                return GenerateSingleFrame(reqPtr);
        }
        
        public static void StopSingleFrameGeneration(int taskID)
        {
            //StopGenerateSingleFrame(taskID);
        }

        public static void StartMultiFrameRecord(MultiFrameRequestOption opt, UnityRuntimeData runtimeData)
        {
            TaskController.runtimeData = runtimeData;

        }

        public static void RecordOneFrame()
        {
        }

        public static void StopMultiFrameRecord(int taskID)
        {
        }

        public static int StartMultiFrameGeneration()
        {
            return -1;
        }

        public static void StopMultiFrameGeneration(int taskID)
        {
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
