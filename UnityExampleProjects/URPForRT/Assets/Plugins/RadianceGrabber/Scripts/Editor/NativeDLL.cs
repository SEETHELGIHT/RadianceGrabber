namespace RadGrab
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using UnityEditor;
    using UnityEngine;

    public static class NativeDLL
    {
        public static T Invoke<T, T2>(IntPtr library, params object[] pars)
        {
            IntPtr funcPtr = GetProcAddress(library, typeof(T2).Name);
            if (funcPtr == IntPtr.Zero || funcPtr == null)
            {
                Debug.LogWarning("Could not gain reference to method address.");
                return default(T);
            }

            var func = Marshal.GetDelegateForFunctionPointer(GetProcAddress(library, typeof(T2).Name), typeof(T2));
            return (T)func.DynamicInvoke(pars);
        }

        public static void Invoke<T>(IntPtr library, params object[] pars)
        {
            IntPtr funcPtr = GetProcAddress(library, typeof(T).Name);
            if (funcPtr == IntPtr.Zero || funcPtr == null)
            {
                Debug.LogWarning("Could not gain reference to method address.");
                return;
            }

            var func = Marshal.GetDelegateForFunctionPointer(funcPtr, typeof(T));
            func.DynamicInvoke(pars);
        }

        [DllImport("kernel32", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool FreeLibrary(IntPtr hModule);

        [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32")]
        public static extern IntPtr GetProcAddress(IntPtr hModule, string procedureName);
    }

    public unsafe static class RadGrabberDLL
    {
        static RadGrabberDLL()
        {
            AssemblyReloadEvents.beforeAssemblyReload += DestroyDLL;
        }

        private delegate int GenerateTest(UnityFrameRequest* req);

        private delegate int GenerateSingleFrame(UnityFrameRequest* req);
        private delegate int GenerateSingleFrameIncremental(UnityFrameRequest* req);
        private delegate int SaveSingleFrameRequest([MarshalAs(UnmanagedType.LPStr)] string fileName, UnityFrameRequest* req);
        private delegate int StopGenerateSingleFrame();
        private delegate int IsSingleFrameGenerating();

        private static string RadGrabberDLLNameKey = "radiancegrabber_dllname_key";
        private static string RadGrabberDLLNameDefault = "../../x64/Release/RadianceGrabber";

        public static IntPtr radianceGrabberPtr = IntPtr.Zero;

        public static bool IsRadianceGrabberLoaded() { return radianceGrabberPtr != IntPtr.Zero; }

        public static IntPtr GetRadGrabberDLLPtr()
        {
            if (radianceGrabberPtr == IntPtr.Zero)
                radianceGrabberPtr = NativeDLL.LoadLibrary(EditorPrefs.GetString(RadGrabberDLLNameKey, RadGrabberDLLNameDefault));

            return radianceGrabberPtr;
        }

        public static void LoadDLL()
        {
            radianceGrabberPtr = NativeDLL.LoadLibrary(EditorPrefs.GetString(RadGrabberDLLNameKey, RadGrabberDLLNameDefault));
        }

        public static void DestroyDLL()
        {
            if (radianceGrabberPtr != IntPtr.Zero)
            {
                Debug.Log("Destroy RadiacneGrabber.dll!");
                NativeDLL.FreeLibrary(radianceGrabberPtr);
                radianceGrabberPtr = IntPtr.Zero;
            }
        }

        public static int GenerateTestImpl(UnityFrameRequest* reqPtr)
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;
            }

            return NativeDLL.Invoke<int, GenerateTest>(radianceGrabberPtr, new IntPtr(reqPtr));
        }

        public static int SaveSingleFrameRequestImpl(string fileName, UnityFrameRequest* reqPtr)
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;
            }

            return NativeDLL.Invoke<int, SaveSingleFrameRequest>(radianceGrabberPtr, fileName, new IntPtr(reqPtr));
        }

        public static int GenerateSingleFrameImpl(UnityFrameRequest* reqPtr)
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;
            }

            return NativeDLL.Invoke<int, GenerateSingleFrame>(radianceGrabberPtr, new IntPtr(reqPtr));
        }

        public static int GenerateSingleFrameIncrementalImpl(UnityFrameRequest* reqPtr)
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;
            }

            return NativeDLL.Invoke<int, GenerateSingleFrameIncremental>(radianceGrabberPtr, new IntPtr(reqPtr));
        }

        public static int StopGenerateSingleFrameImpl()
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;
            }

            return NativeDLL.Invoke<int, StopGenerateSingleFrame>(radianceGrabberPtr);
        }

        public static int IsSingleFrameGeneratingImpl()
        {
            if (radianceGrabberPtr == null)
            {
                Debug.LogWarning("RadianceGrabber.dll cannot be loaded..");
                return -1;

            }

            return NativeDLL.Invoke<int, IsSingleFrameGenerating>(radianceGrabberPtr);
        }
    }
}
