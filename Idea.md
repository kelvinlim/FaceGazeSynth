

I am trying to create simulated faces where the eyes are directed in different directions staring with center and then 5, 10, 15, 20 and 30 degrees from center to left or right.  I've run into the problem that the ai generated ones are not realistic. I would like to start with first principles.  The overall goal is to be able to generate synthetic faces with different speciifed eyegazes from center (0 deg) to 30 deg (left or right).

I think this should be done in a phased manner.

1. Start with a physics based eyeball model so we can get the physics of the eyeball and iris correct.  Two solitary eyeballs in space that look center and at different degrees offset.
2. Have a generic neutral face that matches and fits the eyeball into the face.
3. Different synthetic faces with different races and calibrated levels of emotionss (angry, sad, happy, scornful, etc)
   Can you think about this and give me a plan, emphasizing step 1

=====

This is a really well-structured problem, and your instinct to start from physics is exactly right. The reason AI-generated gaze looks wrong is almost always because it skips the optics — particularly  **corneal refraction** , which is the single most important (and most frequently ignored) physical effect in eye rendering.

Let me walk through my thinking on the full plan, then go deep on Phase 1.

---

**The core insight for Phase 1:** An eyeball is *not* a painted sphere. When you look at someone's iris, you're seeing it *through* the cornea, which is a convex refracting surface. This means the iris appears slightly magnified and shifted toward the viewer. When the eye rotates to look off-center, the iris doesn't just slide across the surface — the refraction geometry changes asymmetrically, producing subtle distortions that your visual system is incredibly sensitive to. Getting this wrong is why synthetic eyes look "dead."

**Phase 1 — Physics-Based Eyeball Model**

The goal here is two floating eyeballs, correctly lit, that you can rotate to precise gaze angles and get physically accurate iris appearance. Here's how I'd break it down:

*Anatomical geometry.* Model the eyeball as a composite of a main sphere (sclera, ~24mm diameter) with a protruding smaller-radius dome (cornea, ~7.8mm radius of curvature, protruding ~2.5mm). Inside, place the iris as a flat annular disc (roughly 12mm outer diameter) set back ~3.6mm behind the corneal apex, with a circular pupil aperture (variable, say 3–4mm for normal lighting). The limbus (cornea-sclera boundary) needs a subtle darkened ring. The sclera isn't pure white — it has slight vascular tinting and subsurface scattering.

*Corneal refraction.* This is the critical physics. The cornea has a refractive index of about 1.376. When rendering, any ray from the camera that hits the cornea must be refracted (Snell's law at the curved surface) before intersecting the iris plane. This produces a virtual image of the iris that is magnified roughly 4–8% and shifted forward. At off-axis gaze angles, this refraction becomes asymmetric — the near edge of the iris is magnified differently than the far edge. You need to implement this correctly, either via ray tracing or a good approximation in a shader.

*Gaze rotation mechanics.* The eye rotates around a center approximately 13.5mm behind the corneal apex. For a given gaze angle θ (your 0°, 5°, 10°, 15°, 20°, 30° targets), you rotate the entire eyeball geometry around this point. The key subtlety: as the eye rotates, the visible sclera area becomes asymmetric (more white on one side), and the foreshortened iris shape changes through the refracting cornea — this is what makes gaze direction readable to an observer.

*Materials and lighting.* The cornea needs a strong specular reflection (the bright highlight you see on eyes is a Purkinje image on the corneal surface). The iris needs radial fibrous texture with some depth. The sclera needs slight translucency. A single key light plus ambient is enough to start, but the specular highlight position matters — it's one of the cues humans use to judge gaze direction.

*Implementation approach.* I'd recommend Python with one of two paths:

* **Path A (recommended for accuracy):** Build a custom ray tracer focused on the eye. You don't need a general-purpose renderer — just sphere/dome intersection, Snell's law refraction, and a flat iris plane with texture. Libraries like `numpy` for math, `PIL` or `matplotlib` for output. This gives you full control over the optics and is the most educational/debuggable path.
* **Path B (faster to photorealism):** Use Blender's Python API (`bpy`) to build the geometry with proper materials (principled BSDF for sclera/iris, glass BSDF for cornea with IOR 1.376, glossy for the tear film). Blender's Cycles renderer handles refraction natively. You script gaze angles programmatically.

Either way, the deliverable is: render pairs of eyeballs at 0°, ±5°, ±10°, ±15°, ±20°, ±30° and validate that the iris displacement follows the expected physical curve (which is *not* linear due to the refraction).

*Validation.* Compare your rendered iris position at each angle against published data on apparent iris displacement vs. gaze angle. Daugman and others have characterized this relationship. If your model matches the empirical curve, the physics is right.

**Phase 2 — Face Integration (brief outline)**

Once the eyeballs render correctly, you embed them into a parametric face mesh. The key challenges are: eyelid geometry that conforms to the eyeball curvature (lids follow the globe), realistic eye socket shadowing, and tear film / caruncle details. A FLAME or Basel Face Model mesh gives you a parametric neutral face you can fit around the calibrated eye model. The eyes should be placed at anatomically correct interpupillary distance (~63mm average) and depth in the socket.

**Phase 3 — Diverse Faces with Emotions (brief outline)**

FLAME and similar models support identity parameters (for demographic variation) and expression parameters (for emotions). You'd map emotion labels to specific blendshape coefficient ranges, validated against FACS (Facial Action Coding System) — e.g., "happy" maps to AU6+AU12. Racial diversity comes from varying shape, texture, and skin reflectance parameters. The critical constraint: eye gaze must remain calibrated through all of this, which means the eyeball model from Phase 1 stays fixed and the face deforms *around* it.

---

Want me to start building the Phase 1 ray tracer? I'd begin with a single eyeball rendering at 0° gaze, get the corneal refraction right, then extend to angle sweeps. That would give you a concrete foundation to validate before moving forward.
