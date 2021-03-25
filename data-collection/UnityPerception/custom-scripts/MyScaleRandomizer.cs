using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers.SampleRandomizers.Tags;
using UnityEngine.Perception.Randomization.Samplers;

namespace UnityEngine.Perception.Randomization.Randomizers.SampleRandomizers
{

    [Serializable]
    [AddRandomizerMenu("Perception/My Scale Randomizer")]
    public class MyScaleRandomizer : Randomizer
    {
        /// <summary>
        /// Randomizes the scale of objects tagged with a MyScaleRandomizerTag
        /// </summary>
        protected override void OnIterationStart()
        {
            var tags = tagManager.Query<RotationRandomizerTag>();
            
            foreach (var tag in tags)
            {

                // Uniform distribution
                float scaleFactor = UnityEngine.Random.Range(0.5f, 2.5f);

                // Constant scale factor
                //float scaleFactor = 1.5f;

                // Random distribution by adding up of uniforms
                // float sampleMean = 1.7f;
                // float scaleFactor = Math.Abs(10.0f * UnityEngine.Random.value - 5.0f + sampleMean);
                tag.transform.localScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);
            }
        }
    }
}
