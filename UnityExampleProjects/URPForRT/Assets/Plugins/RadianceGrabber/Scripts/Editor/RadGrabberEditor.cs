namespace RadGrab
{
    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;
    using UnityEditor;
    using System;
    using System.Runtime.InteropServices;
    using System.Linq;
    using UnityEditor.SceneManagement;
    using System.IO;

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
                cams = Resources.FindObjectsOfTypeAll<Camera>().Where(              obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy && obj.enabled && obj.gameObject.activeInHierarchy).ToArray(),
                smrs = Resources.FindObjectsOfTypeAll<SkinnedMeshRenderer>().Where( obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy && obj.enabled && obj.gameObject.activeInHierarchy).ToArray(),
                mrs = Resources.FindObjectsOfTypeAll<MeshRenderer>().Where(         obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy && obj.enabled && obj.gameObject.activeInHierarchy).ToArray(),
                ls = Resources.FindObjectsOfTypeAll<Light>().Where(                 obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy && obj.enabled && obj.gameObject.activeInHierarchy).ToArray(),
                txs = Resources.FindObjectsOfTypeAll<Texture>().Where(              obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy /*&& obj.enabled && obj.gameObject.activeInHierarchy*/).ToArray(),
                sbs = Resources.FindObjectsOfTypeAll<Skybox>().Where(               obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy && obj.enabled && obj.gameObject.activeInHierarchy).ToArray()
            };
        }

        public static Camera[] GetCameras()
        {
            return Resources.FindObjectsOfTypeAll<Camera>().Where(obj => (obj.hideFlags & HideFlags.HideInHierarchy) != HideFlags.HideInHierarchy).ToArray();
        }
    }

    public static class RadGrabberEditorValue
    {
        private static string textureSizeKeyW = "radiancegrabber_texturesize_width";
        private static string textureSizeKeyH = "radiancegrabber_texturesize_height";
        private static string samplingCountKey = "radiancegrabber_samplingcount";
        private static string maxSamplingCountKey = "radiancegrabber_maxsamplingcount";
        private static string maxPathDepthKey = "radiancegrabber_maxpathdepth";
        private static string selectedCameraKey = "radiancegrabber_selectedcameraindex";
        private static string autoGenerateKey = "radiancegrabber_autogenerate";
        private static string foldPathIntgtrConfigKey = "radiancegrabber_fold_pathintgrtr_cofing";
        private static string foldMiscellConfigKey = "radiancegrabber_fold_miscellaneous_cofing";
        private static string dllPathkey = "radiancegrabber_dllpath";

        public static Vector2Int GetTextureSize() { return new Vector2Int(EditorPrefs.GetInt(textureSizeKeyW, 1024), EditorPrefs.GetInt(textureSizeKeyH, 768)); }
        public static void SetTextureSize(Vector2Int size) { EditorPrefs.SetInt(textureSizeKeyW, size.x); EditorPrefs.SetInt(textureSizeKeyH, size.y); }

        public static int GetSamplingCount() { return EditorPrefs.GetInt(samplingCountKey, 1); }
        public static void SetSamplingCount(int samplingCount) { EditorPrefs.SetInt(samplingCountKey, samplingCount); }

        public static int GetMaxSamplingCount() { return EditorPrefs.GetInt(maxSamplingCountKey, 500); }
        public static void SetMaxSamplingCount(int samplingCount) { EditorPrefs.SetInt(maxSamplingCountKey, samplingCount); }

        public static int GetMaxPathDepth() { return EditorPrefs.GetInt(maxPathDepthKey, 50); }
        public static void SetMaxPathDepth(int depth) { EditorPrefs.SetInt(maxPathDepthKey, depth); }

        public static int GetCameraID() { return EditorPrefs.GetInt(selectedCameraKey); } 
        public static void SetCameraID(int idx) { EditorPrefs.SetInt(selectedCameraKey, idx); }

        public static bool GetAutoGenerate() { return EditorPrefs.GetBool(autoGenerateKey, false); }
        public static void SetAutoGenreeate(bool auto) { EditorPrefs.SetBool(autoGenerateKey, auto); }

        public static bool GetFoldPathIntegratorConfig() { return EditorPrefs.GetBool(foldPathIntgtrConfigKey, true); } 
        public static void SetFoldPathIntegratorConfig(bool fold) { EditorPrefs.SetBool(foldPathIntgtrConfigKey, fold); }

        public static bool GetFoldMiscellaneousConfig() { return EditorPrefs.GetBool(foldMiscellConfigKey, true); }
        public static void SetFoldMiscellaneousConfig(bool fold) { EditorPrefs.SetBool(foldMiscellConfigKey, fold); }

        public static string GetDLLPath() { return EditorPrefs.GetString(dllPathkey, "../../x64/Release/RadianceGrabber"); } 
        public static void SetDLLPath(string path) { EditorPrefs.SetString(dllPathkey, path); }
    }

    public class RadGrabberConfigWindow : EditorWindow
    {
        [MenuItem("Window/RadGrabber/Configuration", priority = 2)]
        [MenuItem("RadGrabber/Configuration", priority = 2)]
        private static void LoadRadianceGrabberWindow()
        {
            RadGrabberConfigWindow wnd
                = EditorWindow.GetWindow<RadGrabberConfigWindow>(false, "Configuration", true);
        }

        private void OnEnable()
        {
            titleContent = new GUIContent("Configuration");
        }

        private void OnGUI()
        {
            RadGrabberEditorValue.SetFoldPathIntegratorConfig(
                EditorGUILayout.BeginFoldoutHeaderGroup(RadGrabberEditorValue.GetFoldPathIntegratorConfig(), "PathIntegrator")
            );

            if (RadGrabberEditorValue.GetFoldPathIntegratorConfig())
            {
                EditorGUI.indentLevel++;
                RadGrabberEditorValue.SetMaxSamplingCount(EditorGUILayout.DelayedIntField("Max Sampling Count", RadGrabberEditorValue.GetMaxSamplingCount()));
                RadGrabberEditorValue.SetMaxPathDepth(EditorGUILayout.DelayedIntField("Max Depth", RadGrabberEditorValue.GetMaxPathDepth()));

                if (GUILayout.Button("Set to Default Values"))
                {
                    RadGrabberEditorValue.SetMaxSamplingCount(500);
                    RadGrabberEditorValue.SetMaxPathDepth(50);
                }
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndFoldoutHeaderGroup();

            RadGrabberEditorValue.SetFoldMiscellaneousConfig(
                EditorGUILayout.BeginFoldoutHeaderGroup(RadGrabberEditorValue.GetFoldMiscellaneousConfig(), "Miscellaneous")
            );

            if (RadGrabberEditorValue.GetFoldMiscellaneousConfig())
            {
                EditorGUI.indentLevel++;
                RadGrabberEditorValue.SetTextureSize(
                    EditorGUILayout.Vector2IntField(
                            "Resolution", 
                            RadGrabberEditorValue.GetTextureSize()
                        )
                    );
                RadGrabberEditorValue.SetAutoGenreeate(
                    EditorGUILayout.Toggle(
                            "Auto restart when changed", 
                            RadGrabberEditorValue.GetAutoGenerate()
                        )
                    );
                RadGrabberEditorValue.SetDLLPath(
                    EditorGUILayout.DelayedTextField(
                            "DLL Path",
                            RadGrabberEditorValue.GetDLLPath()
                        )
                    );
                EditorGUILayout.BeginHorizontal();
                {
                    if (GUILayout.Button("Select DLL Path"))
                    {
                        int index = Application.dataPath.LastIndexOf("Assets");
                        var projectPath = Application.dataPath.Remove(index);
                        string filePath = EditorUtility.OpenFilePanelWithFilters("", projectPath, new string[] { "Dynamic Link Library", "dll" });

                        if (!String.IsNullOrEmpty(filePath))
                        {
                            var fileUri = new Uri(filePath);
                            var refUri = new Uri(projectPath);
                            var relativeUri = Uri.UnescapeDataString(refUri.MakeRelativeUri(fileUri).ToString());
                            RadGrabberEditorValue.SetDLLPath(relativeUri);
                        }
                    }
                }
                EditorGUILayout.EndHorizontal();
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndFoldoutHeaderGroup();

            if (TaskController.IsGeneratingSingleFrame())
            {
                EditorGUILayout.HelpBox("Changed config applied after processing", MessageType.Warning, true);
            }
        }
    }

    public class RadGrabberPreviewData
    {
        public bool processComplete;
        public Texture2D previewTexture;
        public int maxSamplingCount;
        public int maxPathDepth;
    }

    public class RadGrabberPreviewWindow : EditorWindow, IHasCustomMenu
    {
        private static int samplingCount = 0;
        private static int prevSamplingCount = 0;

        private static RadGrabberPreviewData previewData = new RadGrabberPreviewData();
        private static Texture2D textureBuffer = null;
        private bool initialized = false;

        [DllImport("RadianceGrabber", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private static extern void FlushLog();

        [MenuItem("Window/RadGrabber/Preview", priority =1)]
        [MenuItem("RadGrabber/Preview", priority=1)]
        private static void LoadRadianceGrabberWindow()
        {
            RadGrabberPreviewWindow wnd = EditorWindow.GetWindow<RadGrabberPreviewWindow>(false, "Preview", true);
        }

        [MenuItem("RadGrabber/Load RadianceGrabber", priority = 1000)]
        private static void LoadRadianceGrabber() 
        {
            RadGrabberDLL.LoadDLL();
            TaskController.Initialize();
            prevSamplingCount = samplingCount = 0;
        }    
        [MenuItem("RadGrabber/Load RadianceGrabber", priority = 1000, validate = true)]
        private static bool IsLoadRadianceGrabberEnable()
        {
            return !RadGrabberDLL.IsRadianceGrabberLoaded();
        }

        [MenuItem("RadGrabber/Free RadianceGrabber", priority = 1001)]
        private static void FreeRadianceGrabber()
        {
            TaskController.Clear();
            RadGrabberDLL.DestroyDLL();
        }
        [MenuItem("RadGrabber/Free RadianceGrabber", priority = 1001, validate = true)]
        private static bool IsFreeRadianceGrabberEnable()
        {
            return RadGrabberDLL.IsRadianceGrabberLoaded();
        }

        #region Output Texture
        private static Texture2D CreateTexture()
        {
            Vector2Int textureSize = RadGrabberEditorValue.GetTextureSize();
            return new Texture2D(textureSize.x, textureSize.y, TextureFormat.RGBA32, 0, true);
        }
        
        public Texture2D GetTextureBuffer()
        {
            return previewData.previewTexture;
        }

        public Texture2D RefreshTextureSize(Vector2Int size)
        {
            textureBuffer.Resize(size.x, size.y);
            return textureBuffer;
        }
        #endregion

        private void OnTextureBuffer(Vector2 pos, Vector2 size)
        {
            Texture tex = GetTextureBuffer();

            if (tex == null) return;

            Vector2Int clampedSize;
            Vector2Int offset;
            float aspect = (float)tex.width / tex.height;

            if (size.x < size.y * aspect)
            {
                clampedSize = new Vector2Int((int)size.x, (int)(size.x / aspect));
                offset = new Vector2Int(0, (int)(size.y - clampedSize.y) / 2);
            }
            else
            {
                clampedSize = new Vector2Int((int)(size.y * aspect), (int)size.y);
                offset = new Vector2Int((int)(size.x - clampedSize.x) / 2, 0);
            }

            Rect rect = new Rect(pos + offset, clampedSize);
            EditorGUI.DrawPreviewTexture(rect, tex);
        }

        private void SetSamplingCountForTrigger(int samplingCount)
        {
            RadGrabberPreviewWindow.samplingCount = samplingCount;
        }

        private void OnHierarchyUpdate()
        {
            if (RadGrabberEditorValue.GetAutoGenerate())
            {
                TaskController.StopSingleFrameGeneration();
                StartSingleFrameGenerationByPrefs(SetSamplingCountForTrigger);
            }
        }

        private void OnEnable()
        {
            if (previewData.previewTexture == null)
            {
                Vector2Int res = RadGrabberEditorValue.GetTextureSize();
                previewData.previewTexture = new Texture2D(res.x, res.y, TextureFormat.RGBAFloat, false);
            }

            titleContent = new GUIContent("Preview");
            EditorApplication.hierarchyChanged += OnHierarchyUpdate;

            if (!initialized)
            {
                initialized = true;

                if (RadGrabberEditorValue.GetAutoGenerate() && RadGrabberDLL.IsRadianceGrabberLoaded())
                    StartSingleFrameGenerationByPrefs(SetSamplingCountForTrigger);
            }
        }

        private void Update()
        {
            if (TaskController.IsGeneratingSingleFrame())
            {
                if (prevSamplingCount != samplingCount)
                {
                    Debug.LogFormat("Sampling Count Update :: {0}", samplingCount);

                    prevSamplingCount = samplingCount;
                }

                TaskController.UploadColorFromNativeArray();
                Repaint();
            }
        }

        private void OnDisable()
        {
            EditorApplication.hierarchyChanged -= OnHierarchyUpdate;
        }

        private void OnGUI()
        {
            if (!RadGrabberDLL.IsRadianceGrabberLoaded())
            {
                if (GUILayout.Button("Load RadianceGrabber"))
                {
                    LoadRadianceGrabber();
                }
                return;
            }

            Vector2Int textureSize = RadGrabberEditorValue.GetTextureSize();
            bool generating = TaskController.IsGeneratingSingleFrame(), buttonPressed = false, autoGeneratePressed = false;
            float height = 35, padding = 7, intFieldSize = 70;
            int selectedIndex = 0;

            EditorGUI.BeginDisabledGroup(generating);
            {
                EditorGUI.BeginChangeCheck();

                GUIContent content; 
                Vector2 pos = new Vector2(padding, padding), size;

                bool value = RadGrabberEditorValue.GetAutoGenerate();
                EditorGUI.BeginChangeCheck();
                {
                    content = new GUIContent("Auto");
                    size = EditorStyles.toggle.CalcSize(content);
                    value = EditorGUI.ToggleLeft(new Rect(pos, size), content, value);
                    pos.x += size.x + padding;
                }
                if (EditorGUI.EndChangeCheck())
                {
                    RadGrabberEditorValue.SetAutoGenreeate(value);
                    autoGeneratePressed = value;
                }
                
                content = new GUIContent("Image size:");
                size = EditorStyles.numberField.CalcSize(content);
                EditorGUI.LabelField(new Rect(pos, size), content);
                pos.x += size.x + padding;

                textureSize.x = EditorGUI.DelayedIntField(new Rect(pos, new Vector2(intFieldSize, size.y)), textureSize.x);
                pos.x += intFieldSize + padding;

                textureSize.y = EditorGUI.DelayedIntField(new Rect(pos, new Vector2(intFieldSize, size.y)), textureSize.y);
                pos.x += intFieldSize + padding;

                if (EditorGUI.EndChangeCheck())
                {
                    RadGrabberEditorValue.SetTextureSize(textureSize);
                    textureBuffer = RefreshTextureSize(textureSize);
                    TaskController.StopSingleFrameGeneration();
                }
                
                EditorGUI.BeginChangeCheck();

                Camera[] cams = UnityRuntimeData.GetCameras();
                GUIContent[] camNames = Array.ConvertAll(cams, (c) => new GUIContent(c.gameObject.name));
                size = Vector2.zero;
                Array.ForEach(camNames, (cn) => { size = Vector2.Max(size, EditorStyles.popup.CalcSize(cn)); });
                selectedIndex = Array.FindIndex(cams, (c) => c.GetInstanceID() == RadGrabberEditorValue.GetCameraID());
                selectedIndex = Mathf.Clamp(selectedIndex, 0, camNames.Length - 1);

                EditorGUI.BeginDisabledGroup(cams.Length == 0);
                selectedIndex = EditorGUI.Popup(new Rect(pos, size), selectedIndex, camNames);
                pos.x += size.x + padding;

                content = new GUIContent("GO");
                size = GUI.skin.button.CalcSize(content);
                if (GUI.Button(new Rect(pos, size), content))
                    Selection.activeGameObject = cams[selectedIndex].gameObject;
                pos.x += size.x + padding;
                EditorGUI.EndDisabledGroup();

                if (EditorGUI.EndChangeCheck())
                {
                    RadGrabberEditorValue.SetCameraID(cams[selectedIndex].GetInstanceID());
                }

                if (generating)
                {
                    content = new GUIContent(String.Format("{0}/{1}", samplingCount, RadGrabberEditorValue.GetMaxSamplingCount()));
                    size = EditorStyles.label.CalcSize(content);
                    EditorGUI.LabelField(new Rect(pos, size), content);
                    pos.x += size.x + padding;
                }
            }
            EditorGUI.EndDisabledGroup();

            if (!generating)
            {
                GUIStyle btnStyle = GUI.skin.button;
                GUIContent content = new GUIContent("Generate");
                Vector2 btnSize = btnStyle.CalcSize(content);
                buttonPressed = GUI.Button(new Rect(new Vector2(position.size.x - btnSize.x - padding, padding), btnSize), content, btnStyle);
            }
            else
            {
                GUIStyle btnStyle = GUI.skin.button;
                GUIContent content = new GUIContent("Stop");
                Vector2 btnSize = btnStyle.CalcSize(content);
                buttonPressed = GUI.Button(new Rect(new Vector2(position.size.x - btnSize.x - padding, padding), btnSize), content, btnStyle);
            }

            if ((buttonPressed && !generating) || autoGeneratePressed)
            {
                StartSingleFrameGeneration(SetSamplingCountForTrigger, textureSize, UnityRuntimeData.GetCollectedData(), selectedIndex);
            }
            else if (buttonPressed && generating)
            {
                TaskController.StopSingleFrameGeneration();
            }

            OnTextureBuffer(new Vector2(0, height), new Vector2(position.size.x, position.size.y - height - padding));
        }

        [ContextMenu("Save current request")]
        private void SaveCurrentRequest()
        {
            FrameRequestOption opt = new FrameRequestOption()
            {
                resultImageResolution = RadGrabberEditorValue.GetTextureSize(),
                maxSamplingCount = RadGrabberEditorValue.GetMaxSamplingCount(),
                maxDepth = RadGrabberEditorValue.GetMaxPathDepth(),
                updateFuncPtr = IntPtr.Zero
            };
            UnityRuntimeData data = UnityRuntimeData.GetCollectedData();
            int index = Mathf.Clamp(Array.FindIndex(data.cams, (c) => c.GetInstanceID() == RadGrabberEditorValue.GetCameraID()), 0, data.cams.Length);

            TaskController.SaveSingleFrame(EditorSceneManager.GetActiveScene().name, opt, data, index);
        }

        [ContextMenu("Save processed frame")]
        private void SaveProcessedFrame()
        {
            string[] extensions = { "JPEG", "jpg", "PNG", "png", "Truevision TGA", "tga" };
            string  folderPath = EditorUtility.OpenFolderPanel("", Application.dataPath.Replace("/Assets", ""), ""),
                    filePath = String.Format("{0}/{1}.png", folderPath, EditorSceneManager.GetActiveScene().name);

            if (String.IsNullOrEmpty(folderPath))
                return;

            byte[] pngBytes = previewData.previewTexture.EncodeToPNG();
            FileStream stream = File.OpenWrite(filePath);
            stream.Write(pngBytes, 0, pngBytes.Length);
            stream.Close();

            System.Diagnostics.Process.Start(filePath);
        }

        private void StartSingleFrameGenerationByPrefs(Action<int> samplingAction)
        {
            FrameRequestOption opt = new FrameRequestOption()
            {
                resultImageResolution = RadGrabberEditorValue.GetTextureSize(),
                maxSamplingCount = RadGrabberEditorValue.GetMaxSamplingCount(),
                maxDepth = RadGrabberEditorValue.GetMaxPathDepth(),
                updateFuncPtr = Marshal.GetFunctionPointerForDelegate(samplingAction)
            };
            UnityRuntimeData data = UnityRuntimeData.GetCollectedData();
            int index = Mathf.Clamp(Array.FindIndex(data.cams, (c) => c.GetInstanceID() == RadGrabberEditorValue.GetCameraID()), 0, data.cams.Length);

            previewData.processComplete = false;
            if (previewData.previewTexture == null)
                previewData.previewTexture = new Texture2D(opt.resultImageResolution.x, opt.resultImageResolution.y, TextureFormat.RGBAFloat, false);
            else
                previewData.previewTexture.Resize(opt.resultImageResolution.x, opt.resultImageResolution.y);
            previewData.maxSamplingCount = RadGrabberEditorValue.GetMaxSamplingCount();
            previewData.maxPathDepth = RadGrabberEditorValue.GetMaxPathDepth();

            TaskController.result = previewData.previewTexture;
            int code;
            if ((code = TaskController.StartSingleFrameGeneration(opt, data, index)) != 0)
                Debug.LogErrorFormat("Failure :: start single frame generation, code:{0}", code);
            else
                Debug.Log("Successive start single frame generation!");

            EditorWindow wnd = EditorWindow.GetWindow(typeof(RadGrabberConfigWindow));
            wnd.Repaint();
        }

        private static void StartSingleFrameGeneration(Action<int> samplingAction, Vector2Int texSize, UnityRuntimeData data, int selectedIndex)
        {
            FrameRequestOption opt = new FrameRequestOption()
            {
                resultImageResolution = texSize,
                maxSamplingCount = RadGrabberEditorValue.GetMaxSamplingCount(),
                maxDepth = RadGrabberEditorValue.GetMaxPathDepth(),
                updateFuncPtr = Marshal.GetFunctionPointerForDelegate(samplingAction)
            };

            previewData.processComplete = false;
            if (previewData.previewTexture == null)
                previewData.previewTexture = new Texture2D(opt.resultImageResolution.x, opt.resultImageResolution.y, TextureFormat.RGBAFloat, false);
            else
                previewData.previewTexture.Resize(opt.resultImageResolution.x, opt.resultImageResolution.y);
            previewData.maxSamplingCount = RadGrabberEditorValue.GetMaxSamplingCount();
            previewData.maxPathDepth = RadGrabberEditorValue.GetMaxPathDepth();

            TaskController.result = previewData.previewTexture;
            int code;
            if ((code = TaskController.StartSingleFrameGeneration(opt, data, selectedIndex)) != 0)
                Debug.LogErrorFormat("Failure :: start single frame generation, code:{0}", code);
            else
                Debug.Log("Successive start single frame generation!");

            EditorWindow wnd = EditorWindow.GetWindow(typeof(RadGrabberConfigWindow));
            wnd.Repaint();
        }

        public void AddItemsToMenu(GenericMenu menu)
        {
            if (RadGrabberDLL.IsRadianceGrabberLoaded())
            {
                menu.AddDisabledItem(new GUIContent("Load DLL"));
                menu.AddItem(new GUIContent("Destroy DLL"), false, (obj) => { RadGrabberDLL.DestroyDLL(); }, null);
                menu.AddSeparator("");

                if (TaskController.IsGeneratingSingleFrame())
                {
                    menu.AddDisabledItem(new GUIContent("Generate"));
                    menu.AddItem(new GUIContent("Stop"), false, (obj) => { TaskController.StopSingleFrameGeneration(); }, null);
                }
                else
                {
                    menu.AddItem(new GUIContent("Generate"), false, (obj) => { StartSingleFrameGenerationByPrefs(SetSamplingCountForTrigger); }, null);
                    menu.AddDisabledItem(new GUIContent("Stop"));
                }

                menu.AddSeparator("");

                menu.AddItem(new GUIContent("Save current request"), false, (obj) => { SaveCurrentRequest(); }, null);
                menu.AddItem(new GUIContent("Flush log"), false, (obj) => { FlushLog(); }, null);
            }
            else
            {
                menu.AddItem(new GUIContent("Load DLL"), false, (obj) => { RadGrabberDLL.LoadDLL(); }, null);
                menu.AddDisabledItem(new GUIContent("Destroy DLL"));
            }

            menu.AddSeparator("");

            menu.AddItem(new GUIContent("Save processed frame"), false, (obj) => { SaveProcessedFrame(); }, null);
            //menu.AddItem(new GUIContent("Destroy DLL"), false, (obj) => { RadGrabberDLL.DestroyDLL(); }, null);
        }
    }
}
