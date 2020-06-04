using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
[ExecuteInEditMode]
public class  BoundsTest : MonoBehaviour
{
    /*
     *  This implicate extents of Bounds class is half of whole size
     */
    [ContextMenu("Print Bounds on console")]
    private void OnBounds()
    {
        var col = GetComponent<BoxCollider>();
        if (col != null)
            Debug.Log(col.bounds);
    }
}
