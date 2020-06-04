using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TextureCopyTest : MonoBehaviour
{
    [SerializeField]
    private Texture2D targetTexture;

    [ContextMenu("Copy targetTexture")]
    private void OnCopy()
    {
        int x = 0, y = 0;
        Color32[] cs = null;
        RadGrab.Texture2DChunk.CopyTextureData(targetTexture, ref x, ref y, ref cs);
        
        RawImage i = GetComponent<RawImage>();
        Texture2D tex = new Texture2D(x, y, TextureFormat.ARGB32, false, true);
        tex.SetPixels32(cs, 0);
        tex.Apply();
        i.texture = tex; 

        Debug.Log(transform.rotation * new Vector3(0, 0, 1));
    }
}
