using RadGrab;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

[ExecuteInEditMode]
public unsafe class QuatTest : MonoBehaviour
{
    private void OnEnable()
    {
        MeshRenderer mr;
        if (mr = GetComponent<MeshRenderer>())
        {
            Debug.Log(mr.bounds);
        }
        MeshFilter f;
        if ((f = GetComponent<MeshFilter>()) && f.sharedMesh != null)
        {
            Mesh m = f.sharedMesh;
            m.RecalculateNormals();
            m.RecalculateTangents();
            Debug.Log(m.tangents.Length);
            Debug.Log(m.normals.Length);
        }

        Debug.LogFormat("rotation :: {2}, size of camera chunk :: {0}, matrix chunk :: {1}", Marshal.SizeOf<CameraChunk>(), Marshal.SizeOf<Matrix4x4>(), transform.rotation);
    }
}
