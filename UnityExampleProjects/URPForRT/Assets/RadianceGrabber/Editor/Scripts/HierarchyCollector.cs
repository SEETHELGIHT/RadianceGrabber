namespace RadGrab
{
    using System.Collections;
    using System.Collections.Generic;
    using UnityEditor;
    using UnityEngine;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System;
    using UnityEngine.Rendering;

    public struct UnityRuntimeData
    {
        public Camera[] cams;
        public SkinnedMeshRenderer[] smrs;
        public MeshRenderer[] mrs;
        public Light[] ls;
        public Texture[] txs;
        public Skybox[] sbs;

        public static UnityRuntimeData GetCollectedData()
        {
            return new UnityRuntimeData()
            {
                cams = Resources.FindObjectsOfTypeAll<Camera>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray(),
                smrs = Resources.FindObjectsOfTypeAll<SkinnedMeshRenderer>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray(),
                mrs = Resources.FindObjectsOfTypeAll<MeshRenderer>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray(),
                ls = Resources.FindObjectsOfTypeAll<Light>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray(),
                txs = Resources.FindObjectsOfTypeAll<Texture>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray(),
                sbs = Resources.FindObjectsOfTypeAll<Skybox>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray()
            };
        }
    }

    [InitializeOnLoad]
    public static class HierarchyCollector
    {
        static HierarchyCollector()
        {
            EditorApplication.hierarchyChanged += OnHierarchyChanged;
        }

        private static unsafe void OnHierarchyChanged()
        {
            UnityRuntimeData runtimeData = UnityRuntimeData.GetCollectedData();

            Debug.LogFormat("There are currently {0} Camera, {1} SkinnedMeshRenderer, {2} MeshRenderer, {3} Light, {4} Texture, {5} Skybox visible in the hierarchy.", runtimeData.cams.Count(), runtimeData.smrs.Count(), runtimeData.mrs.Count(), runtimeData.ls.Count(), runtimeData.txs.Count(), runtimeData.sbs.Count());

            FrameRequestOption opt = 
                new FrameRequestOption() {
                    resultImageResolution = new Vector2Int(1920, 1080),
                    maxSamplingCount = 500,
                    selectedCameraIndex = 0,
                    updateFuncPtr = Marshal.GetFunctionPointerForDelegate<Action>(() => { Debug.Log("앙 updateFuncPtr 띠"); })
                };

            TaskController.StartSingleFrameGeneration(opt, runtimeData);
        }
    }
}
