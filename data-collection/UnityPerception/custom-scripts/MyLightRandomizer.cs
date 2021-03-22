using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/My Light Randomizer")]


public class MyLightRandomizer : Randomizer
{
    public FloatParameter lightIntensityParameter;
    public ColorRgbParameter lightColorParameter;

    protected override void OnIterationStart()
    {
        var tags = tagManager.Query<MyLightRandomizerTag>();

        foreach (var tag in tags)
        {
            var light = tag.GetComponent<Light>();
            light.intensity = lightIntensityParameter.Sample();
            light.color = lightColorParameter.Sample();

            // these must be modified inside the script, i wrote them 
            light.transform.position         = new Vector3(UnityEngine.Random.Range(-20.0f, 20.0f), UnityEngine.Random.Range(-20.0f, 20.0f), -10.0f);
            light.transform.localEulerAngles = new Vector3(UnityEngine.Random.Range(-70.0f, 70.0f), UnityEngine.Random.Range(-70.0f, 70.0f), 0);
        }

    }
}


