using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode()]
public class DirectionTest : MonoBehaviour
{
    private void OnEnable()
    {
        Debug.Log(transform.forward);
    }
}
