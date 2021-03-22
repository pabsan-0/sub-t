## Unity Perception 
The tool Unity Perception is used to generate virtual images from CAD models that provide extra training data. This folder holds some of assets that were used to generate said synthetic data:

- 3D models: which were imported from the models available at [Ignition Robotics](https://app.ignitionrobotics.org/). These were treated inside the Unity environment to reduce shinyness (for more realistic texture) and fix model issues (for instance the back of the rope, which wasn't initially drawn).
- Model textures: which shouldn't be needed but are included as assets just in case.
- Scripts for randomizing light (copied from the Unity tutorial) and scale (self-written) within the simulation, as well as the tags that the items need to undergo the scripted modifications.

The textures used for the background clutter were not including to save disk space. They were extracted from simulation videos with the tools available in the sub-t/data-collection/picture-overlapping_discontinued/ directory, however any other set of noisy textures can be used to achieve similar data.

More on Unity Perception can be found at the official [Unity Perception sitory](https://github.com/Unity-Technologies/com.unity.perception/). The tutorial that was followed for producing the data can be found, within the same repo, [here](https://github.com/Unity-Technologies/com.unity.perception/blob/master/com.unity.perception/Documentation~/Tutorial/Phase1.md).
